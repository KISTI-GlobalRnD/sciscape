#!/usr/bin/env python3
"""Design the first-pass local_pair_014 pathway-probe contract.

This consumes the bounded wall/pathway-readiness audit and freezes the next
predeclared probe. ``local_pair_014`` is the only positive pathway-probe
candidate. ``local_pair_005`` is retained only as a source/target-collapse
boundary control.

This is a contract design only. It does not run Leiden, execute routes, promote
walls, evaluate quality/cost value, replay full NanoClustering, or claim method
success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_wall_pathway_readiness import (
    DEFAULT_OUTPUT_DIR as DEFAULT_READINESS_DIR,
    GATE_MATRIX_CSV as READINESS_GATE_MATRIX_CSV,
    PAIR_READINESS_ROWS_CSV,
    ROUTE_READINESS_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_contract_gamma1e5_20260604"
)

RULE_ROWS_CSV = "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_rule_rows.csv"
PAIR_ROWS_CSV = "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_pair_rows.csv"
ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_route_plan_rows.csv"
)
CONTROL_GUARD_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_control_guard_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_report.md"

POSITIVE_PAIR_ID = "local_pair_014"
BOUNDARY_PAIR_ID = "local_pair_005"
AUDIT_PAIR_IDS = (POSITIVE_PAIR_ID, BOUNDARY_PAIR_ID)

RUN_STATUS = "designed_nanoclustering_g4_8_first_pass_014_pathway_probe_contract"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_014 pathway-probe contract design "
    "only; reads the bounded readiness audit and predeclares recovery-loop, "
    "direct-only, and boundary-control checks. It does not run Leiden, execute "
    "routes, promote walls, evaluate quality/cost value, replay full "
    "NanoClustering, or claim method success."
)

REQUIRED_MEASUREMENTS = (
    "endpoint_assignment_by_step",
    "endpoint_object_assignment_by_step",
    "direct_edge_retained_all_steps",
    "bridge_fraction_by_step",
    "first_exclusive_target_step",
    "objective_debt_from_start",
    "objective_recovery_from_min",
    "accepted_recovery_after_min",
    "support_incompatibility_by_step",
    "boundary_control_leak_status",
)

ACCEPTANCE_RULES: tuple[dict[str, str], ...] = (
    {
        "rule_id": "P1_scope",
        "rule_group": "scope",
        "rule_question": "Is the positive probe restricted to local_pair_014?",
        "acceptance_requirement": "positive_pair_id == local_pair_014 and boundary_pair_id == local_pair_005",
        "claim_effect": "necessary_scope_guard",
    },
    {
        "rule_id": "P2_object_precondition",
        "rule_group": "precondition",
        "rule_question": "Does local_pair_014 retain clean endpoint-object evidence?",
        "acceptance_requirement": "object_audit_class == clean_symmetric_endpoint_object_candidate",
        "claim_effect": "pathway_probe_entry_only",
    },
    {
        "rule_id": "P3_boundary_control_precondition",
        "rule_group": "precondition",
        "rule_question": "Is local_pair_005 retained as a non-positive boundary control?",
        "acceptance_requirement": "object_audit_class == partial_boundary_source_target_collapse",
        "claim_effect": "blocks_false_positive_generalization",
    },
    {
        "rule_id": "P4_independent_direct_path_availability",
        "rule_group": "pathway",
        "rule_question": "Can the direct path be accepted independently of bridge-release replay?",
        "acceptance_requirement": (
            "direct-only probe keeps direct edge, suppresses bridge support, reaches an "
            "exclusive target object for all seeds and starts, and has no support incompatibility"
        ),
        "claim_effect": "required_before_wall_readiness",
    },
    {
        "rule_id": "P5_accepted_recovery",
        "rule_group": "wall_shape",
        "rule_question": "Does a recovery-loop schedule show accepted objective recovery after the debt minimum?",
        "acceptance_requirement": (
            "recovery-loop probe reports max_objective_recovery_from_min > 0 after "
            "the minimum while endpoint objects remain interpretable"
        ),
        "claim_effect": "required_before_wall_language",
    },
    {
        "rule_id": "P6_boundary_no_leak",
        "rule_group": "control",
        "rule_question": "Does the 005 boundary control stay non-positive under the same probe families?",
        "acceptance_requirement": "no 005 route is counted as positive pathway evidence",
        "claim_effect": "required_false_positive_guard",
    },
    {
        "rule_id": "P7_wall_claim_closed",
        "rule_group": "claim_boundary",
        "rule_question": "Are wall claims closed in this contract?",
        "acceptance_requirement": "wall_claim_allowed_after_contract == false",
        "claim_effect": "wall_promotion_blocked_until_execution_and_acceptance",
    },
)

POSITIVE_ROUTE_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "planned_route_family": "first_pass_014_recovery_loop_probe",
        "route_family_role": "primary_recovery_probe",
        "planned_intervention_schedule": (
            "direct_edge_weight_fraction=1.0 throughout; "
            "bridge_edge_weight_fraction=1.00,0.75,0.50,0.25,0.00,0.25,0.50,0.75,1.00"
        ),
        "expected_endpoint_pattern": (
            "source_object_to_exclusive_target_object_with_recovery_assessment"
        ),
        "acceptance_rule_ids": "P1;P2;P5;P7",
        "runner_support_status": "new_schedule_support_required",
        "probe_question": (
            "Can the clean 014 bridge-release transition show objective recovery "
            "after the bridge support is reintroduced?"
        ),
    },
    {
        "planned_route_family": "first_pass_014_direct_only_target_availability_probe",
        "route_family_role": "independent_direct_path_probe",
        "planned_intervention_schedule": (
            "baseline source-anchor step with direct_edge_weight_fraction=1.0 and "
            "bridge_edge_weight_fraction=1.00, followed by direct-only target test "
            "with direct_edge_weight_fraction=1.0 and bridge_edge_weight_fraction=0.00"
        ),
        "expected_endpoint_pattern": "exclusive_target_object_available_without_bridge_support",
        "acceptance_rule_ids": "P1;P2;P4;P7",
        "runner_support_status": "new_schedule_support_required",
        "probe_question": (
            "Is the target object available with bridge support suppressed while "
            "the direct edge remains physically retained?"
        ),
    },
)

BOUNDARY_ROUTE_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "planned_route_family": "first_pass_005_boundary_recovery_loop_guard",
        "route_family_role": "boundary_recovery_control",
        "planned_intervention_schedule": (
            "direct_edge_weight_fraction=1.0 throughout; "
            "bridge_edge_weight_fraction=1.00,0.75,0.50,0.25,0.00,0.25,0.50,0.75,1.00"
        ),
        "expected_endpoint_pattern": "source_target_collapse_or_mixed_boundary_not_positive",
        "acceptance_rule_ids": "P3;P6;P7",
        "runner_support_status": "new_schedule_support_required",
        "probe_question": (
            "Does the 005 boundary remain non-positive under the same recovery-loop schedule?"
        ),
    },
    {
        "planned_route_family": "first_pass_005_boundary_direct_only_guard",
        "route_family_role": "boundary_direct_path_control",
        "planned_intervention_schedule": (
            "baseline source-anchor step with direct_edge_weight_fraction=1.0 and "
            "bridge_edge_weight_fraction=1.00, followed by direct-only guard test "
            "with direct_edge_weight_fraction=1.0 and bridge_edge_weight_fraction=0.00"
        ),
        "expected_endpoint_pattern": "boundary_target_availability_must_not_promote_positive",
        "acceptance_rule_ids": "P3;P6;P7",
        "runner_support_status": "new_schedule_support_required",
        "probe_question": (
            "Does the 005 boundary avoid becoming positive under direct-only target availability?"
        ),
    },
)


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _rule_rows() -> pd.DataFrame:
    rows = pd.DataFrame(list(ACCEPTANCE_RULES))
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _pair_rows(pair_readiness: pd.DataFrame) -> pd.DataFrame:
    rows = pair_readiness[
        pair_readiness["local_pair_id"].astype(str).isin(AUDIT_PAIR_IDS)
    ].copy()
    rows["contract_pair_role"] = rows["local_pair_id"].astype(str).map(
        {
            POSITIVE_PAIR_ID: "positive_pathway_probe_candidate",
            BOUNDARY_PAIR_ID: "boundary_collapse_control",
        }
    )
    rows["planned_route_family_count"] = rows["local_pair_id"].astype(str).map(
        {
            POSITIVE_PAIR_ID: len(POSITIVE_ROUTE_FAMILIES),
            BOUNDARY_PAIR_ID: len(BOUNDARY_ROUTE_FAMILIES),
        }
    )
    rows["wall_claim_allowed_after_contract"] = False
    rows["method_claim_allowed_after_contract"] = False
    rows["quality_cost_claim_allowed_after_contract"] = False
    rows["required_measurements"] = ";".join(REQUIRED_MEASUREMENTS)
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(["contract_pair_role", "local_pair_id"], kind="mergesort").reset_index(
        drop=True
    )


def _start_rows(route_readiness: pd.DataFrame, *, local_pair_id: str) -> pd.DataFrame:
    scoped = route_readiness[route_readiness["local_pair_id"].astype(str).eq(local_pair_id)]
    rows = (
        scoped.groupby(["local_pair_id", "branch", "start_condition"], sort=False)
        .agg(
            seed_count=("seed", "nunique"),
            route_pathway_readiness_candidate_count=(
                "route_pathway_readiness_candidate",
                "sum",
            ),
            max_objective_debt_from_start=("max_objective_debt_from_start", "max"),
            max_objective_recovery_from_min=("max_objective_recovery_from_min", "max"),
            pathway_shape_classes=("pathway_shape_class", lambda values: ";".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )
    return rows.sort_values("start_condition", kind="mergesort").reset_index(drop=True)


def _route_plan_rows(route_readiness: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positive_starts = _start_rows(route_readiness, local_pair_id=POSITIVE_PAIR_ID)
    boundary_starts = _start_rows(route_readiness, local_pair_id=BOUNDARY_PAIR_ID)
    for start in positive_starts.itertuples(index=False):
        for order, family in enumerate(POSITIVE_ROUTE_FAMILIES, start=1):
            route_contract_id = (
                f"{POSITIVE_PAIR_ID}__{start.start_condition}__{family['planned_route_family']}"
            )
            rows.append(
                {
                    "route_contract_id": route_contract_id,
                    "local_pair_id": POSITIVE_PAIR_ID,
                    "branch": start.branch,
                    "start_condition": start.start_condition,
                    "contract_pair_role": "positive_pathway_probe_candidate",
                    "route_family_order": order,
                    **family,
                    "current_seed_count": int(start.seed_count),
                    "current_route_pathway_readiness_candidate_count": int(
                        start.route_pathway_readiness_candidate_count
                    ),
                    "current_max_objective_debt_from_start": float(
                        start.max_objective_debt_from_start
                    ),
                    "current_max_objective_recovery_from_min": float(
                        start.max_objective_recovery_from_min
                    ),
                    "current_pathway_shape_classes": start.pathway_shape_classes,
                    "new_route_execution_required": True,
                    "counts_as_positive_if_accepted": True,
                    "wall_claim_allowed_after_contract": False,
                    "required_measurements": ";".join(REQUIRED_MEASUREMENTS),
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    for start in boundary_starts.itertuples(index=False):
        for order, family in enumerate(BOUNDARY_ROUTE_FAMILIES, start=1):
            route_contract_id = (
                f"{BOUNDARY_PAIR_ID}__{start.start_condition}__{family['planned_route_family']}"
            )
            rows.append(
                {
                    "route_contract_id": route_contract_id,
                    "local_pair_id": BOUNDARY_PAIR_ID,
                    "branch": start.branch,
                    "start_condition": start.start_condition,
                    "contract_pair_role": "boundary_collapse_control",
                    "route_family_order": order,
                    **family,
                    "current_seed_count": int(start.seed_count),
                    "current_route_pathway_readiness_candidate_count": int(
                        start.route_pathway_readiness_candidate_count
                    ),
                    "current_max_objective_debt_from_start": float(
                        start.max_objective_debt_from_start
                    ),
                    "current_max_objective_recovery_from_min": float(
                        start.max_objective_recovery_from_min
                    ),
                    "current_pathway_shape_classes": start.pathway_shape_classes,
                    "new_route_execution_required": True,
                    "counts_as_positive_if_accepted": False,
                    "wall_claim_allowed_after_contract": False,
                    "required_measurements": ";".join(REQUIRED_MEASUREMENTS),
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["contract_pair_role", "local_pair_id", "start_condition", "route_family_order"],
        kind="mergesort",
    ).reset_index(drop=True)


def _control_guard_rows(route_plan: pd.DataFrame, pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows = route_plan[
        route_plan["contract_pair_role"].astype(str).eq("boundary_collapse_control")
    ].copy()
    rows["control_guard_id"] = rows["route_contract_id"].astype(str)
    rows["control_guard_family"] = "source_target_collapse_boundary"
    rows["positive_leak_signal"] = (
        "boundary route satisfies positive recovery/direct-only acceptance"
    )
    rows["expected_guard_outcome"] = "must_not_count_as_positive_pathway_evidence"
    pair_lookup = pair_rows.set_index("local_pair_id").to_dict("index")
    rows["boundary_pair_readiness_status"] = rows["local_pair_id"].map(
        lambda pair_id: str(pair_lookup.get(str(pair_id), {}).get("pair_readiness_status", ""))
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.reset_index(drop=True)


def _gate_row(
    gate_id: str,
    question: str,
    observed: Any,
    minimum_or_rule: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "observed": observed,
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
    }


def _gate_matrix(
    *,
    readiness_gates: pd.DataFrame,
    pair_rows: pd.DataFrame,
    route_plan: pd.DataFrame,
    control_guards: pd.DataFrame,
) -> pd.DataFrame:
    positive_routes = route_plan[
        route_plan["contract_pair_role"].astype(str).eq("positive_pathway_probe_candidate")
    ]
    boundary_routes = route_plan[
        route_plan["contract_pair_role"].astype(str).eq("boundary_collapse_control")
    ]
    positive_pair = pair_rows[pair_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    boundary_pair = pair_rows[pair_rows["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)]
    rows = [
        _gate_row(
            "G1_upstream_readiness_gates_pass",
            "Did the upstream wall/pathway-readiness audit gates pass?",
            _count_dict(readiness_gates["gate_status"]),
            "all upstream readiness gates pass",
            bool(readiness_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_scope_014_positive_005_boundary",
            "Is the contract scoped to 014 positive and 005 boundary only?",
            pair_rows[["local_pair_id", "contract_pair_role"]].to_dict("records"),
            "exactly two pair rows with fixed roles",
            len(pair_rows) == 2
            and not positive_pair.empty
            and not boundary_pair.empty
            and str(positive_pair.iloc[0]["contract_pair_role"])
            == "positive_pathway_probe_candidate"
            and str(boundary_pair.iloc[0]["contract_pair_role"]) == "boundary_collapse_control",
        ),
        _gate_row(
            "G3_route_families_predeclared",
            "Are positive and boundary route families predeclared?",
            {
                "positive_route_count": int(len(positive_routes)),
                "boundary_route_count": int(len(boundary_routes)),
                "families": sorted(route_plan["planned_route_family"].astype(str).unique().tolist()),
            },
            "8 positive route rows and 8 boundary route rows",
            len(positive_routes) == 8 and len(boundary_routes) == 8,
        ),
        _gate_row(
            "G4_recovery_and_direct_path_rules_required",
            "Does the contract explicitly require independent direct-path and accepted recovery checks?",
            [rule["rule_id"] for rule in ACCEPTANCE_RULES],
            "P4 and P5 present",
            {"P4_independent_direct_path_availability", "P5_accepted_recovery"}.issubset(
                {rule["rule_id"] for rule in ACCEPTANCE_RULES}
            ),
        ),
        _gate_row(
            "G5_boundary_controls_materialized",
            "Are 005 boundary control guards materialized?",
            f"control_guard_rows={len(control_guards)}",
            "8 boundary guard rows",
            len(control_guards) == 8
            and bool(control_guards["counts_as_positive_if_accepted"].eq(False).all()),
        ),
        _gate_row(
            "G6_new_runner_support_declared",
            "Are new schedule families marked as requiring runner support?",
            _count_dict(route_plan["runner_support_status"]),
            "all route rows require new schedule support",
            bool(route_plan["runner_support_status"].astype(str).eq("new_schedule_support_required").all()),
        ),
        _gate_row(
            "G7_wall_method_quality_claims_closed",
            "Are wall, method, and quality/cost claims closed by contract?",
            CLAIM_BOUNDARY,
            "all wall flags false",
            bool(route_plan["wall_claim_allowed_after_contract"].eq(False).all())
            and bool(pair_rows["wall_claim_allowed_after_contract"].eq(False).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    readiness_dir: Path,
    output_dir: Path,
    pair_rows: pd.DataFrame,
    route_plan: pd.DataFrame,
    control_guards: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_summary.v1",
        "status": RUN_STATUS,
        "readiness_dir": str(readiness_dir),
        "output_dir": str(output_dir),
        "pair_row_count": int(len(pair_rows)),
        "route_plan_row_count": int(len(route_plan)),
        "positive_route_plan_row_count": int(
            route_plan["contract_pair_role"].astype(str).eq("positive_pathway_probe_candidate").sum()
        ),
        "boundary_route_plan_row_count": int(
            route_plan["contract_pair_role"].astype(str).eq("boundary_collapse_control").sum()
        ),
        "control_guard_row_count": int(len(control_guards)),
        "planned_route_family_counts": _count_dict(route_plan["planned_route_family"]),
        "runner_support_status_counts": _count_dict(route_plan["runner_support_status"]),
        "acceptance_rule_count": int(len(ACCEPTANCE_RULES)),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"), "gate_id"
        ].tolist(),
        "interpretation": (
            "The next probe is predeclared but not executed. local_pair_014 is "
            "the only positive pathway candidate; local_pair_005 is retained as "
            "a boundary control. Wall claims remain closed until independent "
            "direct-path and accepted-recovery checks are executed and pass."
        ),
        "recommended_next_gate": (
            "Implement runner support for the recovery-loop and direct-only "
            "families, then execute exactly the 16 route-plan rows in this contract."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    visible = frame[cols].head(int(max_rows))
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for row in visible.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, (dict, list, tuple, set)):
                values.append(json.dumps(_json_safe(value), sort_keys=True))
            elif pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    rule_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    route_plan: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 014 Pathway-Probe Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- positive_route_plan_row_count: {summary['positive_route_plan_row_count']}",
        f"- boundary_route_plan_row_count: {summary['boundary_route_plan_row_count']}",
        f"- planned_route_family_counts: {summary['planned_route_family_counts']}",
        f"- runner_support_status_counts: {summary['runner_support_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Acceptance Rules",
        "",
        _markdown_table(
            rule_rows,
            ["rule_id", "rule_group", "rule_question", "acceptance_requirement", "claim_effect"],
            max_rows=20,
        ),
        "",
        "## Pair Rows",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "contract_pair_role",
                "pair_pathway_readiness_candidate",
                "pair_wall_claim_ready",
                "max_objective_debt_from_start",
                "max_objective_recovery_from_min",
                "wall_claim_missing_fields",
            ],
            max_rows=10,
        ),
        "",
        "## Route Plan",
        "",
        _markdown_table(
            route_plan,
            [
                "route_contract_id",
                "local_pair_id",
                "start_condition",
                "contract_pair_role",
                "planned_route_family",
                "route_family_role",
                "planned_intervention_schedule",
                "expected_endpoint_pattern",
                "runner_support_status",
                "counts_as_positive_if_accepted",
            ],
            max_rows=40,
        ),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gates,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
            max_rows=20,
        ),
        "",
        "## Boundary",
        "",
        (
            "This is a design contract. It must not be interpreted as direct-path, "
            "accepted-recovery, wall, method, or quality/cost evidence until the "
            "predeclared route rows are executed and separately audited."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    readiness_dir = Path(args.readiness_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_readiness = _read_csv(readiness_dir / ROUTE_READINESS_ROWS_CSV)
    pair_readiness = _read_csv(readiness_dir / PAIR_READINESS_ROWS_CSV)
    readiness_gates = _read_csv(readiness_dir / READINESS_GATE_MATRIX_CSV)

    rule_rows = _rule_rows()
    pair_rows = _pair_rows(pair_readiness)
    route_plan = _route_plan_rows(route_readiness)
    control_guards = _control_guard_rows(route_plan, pair_rows)
    gates = _gate_matrix(
        readiness_gates=readiness_gates,
        pair_rows=pair_rows,
        route_plan=route_plan,
        control_guards=control_guards,
    )
    summary = _summary(
        readiness_dir=readiness_dir,
        output_dir=output_dir,
        pair_rows=pair_rows,
        route_plan=route_plan,
        control_guards=control_guards,
        gates=gates,
    )

    _write_csv(rule_rows, output_dir / RULE_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(route_plan, output_dir / ROUTE_PLAN_ROWS_CSV)
    _write_csv(control_guards, output_dir / CONTROL_GUARD_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_config.v1",
        "readiness_dir": str(readiness_dir),
        "output_dir": str(output_dir),
        "positive_pair_id": POSITIVE_PAIR_ID,
        "boundary_pair_id": BOUNDARY_PAIR_ID,
        "required_measurements": list(REQUIRED_MEASUREMENTS),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        rule_rows=rule_rows,
        pair_rows=pair_rows,
        route_plan=route_plan,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
