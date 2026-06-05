#!/usr/bin/env python3
"""Design a scoped Stage 2A pathway-probe contract for G4.8.

This consumes the pathway/wall readiness audit and creates a predeclared tiny
probe contract for the two ready pairs only. Controls are retained as
false-positive guards, not broadened into route execution rows. Wall claims stay
closed until route traces and wall-evidence fields are actually materialized.

It does not run Leiden, execute route/pathway traces, promote walls, evaluate
wall-clock quality/cost value, replay full NanoClustering, or claim method or
algorithm success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_READINESS_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_pathway_wall_readiness_audit_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_contract_gamma1e5_20260604"
)

INPUT_UNIT_ROWS_CSV = "nanoclustering_g4_8_pathway_wall_readiness_audit_unit_rows.csv"
INPUT_PAIR_ROWS_CSV = "nanoclustering_g4_8_pathway_wall_readiness_audit_pair_rows.csv"
INPUT_GATE_MATRIX_CSV = "nanoclustering_g4_8_pathway_wall_readiness_audit_gate_matrix.csv"

CANDIDATE_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_scoped_pathway_probe_contract_candidate_pair_rows.csv"
)
PROBE_UNIT_ROWS_CSV = "nanoclustering_g4_8_scoped_pathway_probe_contract_probe_unit_rows.csv"
ROUTE_PLAN_ROWS_CSV = "nanoclustering_g4_8_scoped_pathway_probe_contract_route_plan_rows.csv"
CONTROL_GUARD_ROWS_CSV = (
    "nanoclustering_g4_8_scoped_pathway_probe_contract_control_guard_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_scoped_pathway_probe_contract_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_scoped_pathway_probe_contract_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_scoped_pathway_probe_contract_summary.json"
REPORT_MD = "nanoclustering_g4_8_scoped_pathway_probe_contract_report.md"

ROUTE_FAMILIES = (
    {
        "planned_route_family": "bridge_release_interpolation_probe",
        "route_family_role": "primary_pathway_probe",
        "planned_intervention_schedule": (
            "preserve_direct; bridge_edge_weight_fraction=1.0,0.75,0.50,0.25,0.0"
        ),
        "expected_endpoint_pattern": (
            "original_partial_source_like_to_drop_bridge_target_like_or_boundary"
        ),
        "probe_question": (
            "Does reducing bridge support produce a measured endpoint transition "
            "from source-like partial coassignment toward target-like coassignment?"
        ),
    },
    {
        "planned_route_family": "direct_dependency_collapse_guard",
        "route_family_role": "candidate_internal_control",
        "planned_intervention_schedule": (
            "preserve_bridge; direct_edge_weight_fraction=1.0,0.75,0.50,0.25,0.0"
        ),
        "expected_endpoint_pattern": "direct_suppression_collapses_pair_not_target_pathway",
        "probe_question": (
            "Does removing direct support collapse the source-like state rather "
            "than create a false target pathway?"
        ),
    },
    {
        "planned_route_family": "drop_both_collapse_guard",
        "route_family_role": "candidate_internal_control",
        "planned_intervention_schedule": (
            "direct_edge_weight_fraction=1.0,0.50,0.0; "
            "bridge_edge_weight_fraction=1.0,0.50,0.0 jointly"
        ),
        "expected_endpoint_pattern": "joint_direct_and_bridge_suppression_remains_collapsed",
        "probe_question": (
            "Does removing both direct and bridge support keep the pathway probe "
            "from hallucinating a target endpoint?"
        ),
    },
)

REQUIRED_MEASUREMENTS = (
    "route_trace_rows",
    "objective_value_by_step",
    "objective_debt_from_start",
    "objective_recovery_from_min",
    "endpoint_assignment_by_step",
    "support_distance_by_step",
    "polish_reversion_check",
    "support_incompatibility_check",
)

RUN_STATUS = "designed_nanoclustering_g4_8_scoped_pathway_probe_contract"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 scoped pathway-probe contract design only; reads the "
    "pathway/wall readiness audit and predeclares tiny candidate route-plan rows "
    "for the two ready pairs. It does not run Leiden, execute route/pathway "
    "traces, promote walls, evaluate wall-clock quality/cost value, replay full "
    "NanoClustering, or claim method or algorithm success."
)


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _prefix_stats(prefix: str, values: pd.Series | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
        }
    return {
        f"{prefix}_min": float(array.min()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_mean": float(array.mean()),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for row in frame[cols].itertuples(index=False):
        values: list[str] = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _candidate_units(unit_rows: pd.DataFrame) -> pd.DataFrame:
    rows = unit_rows[_bool_series(unit_rows["pathway_probe_candidate"])].copy()
    rows["stage2a_contract_role"] = "candidate_probe_unit"
    rows["candidate_probe_contract_status"] = "included_in_scoped_pathway_probe_contract"
    rows["route_execution_status"] = "not_executed_contract_only"
    rows["wall_claim_ready_after_contract"] = False
    rows["required_measurements_before_wall_claim"] = ";".join(REQUIRED_MEASUREMENTS)
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(["local_pair_id", "start_condition"], kind="mergesort").reset_index(
        drop=True
    )


def _candidate_pairs(pair_rows: pd.DataFrame, probe_units: pd.DataFrame) -> pd.DataFrame:
    pair_candidates = pair_rows[_bool_series(pair_rows["pathway_probe_candidate_pair"])].copy()
    unit_counts = (
        probe_units.groupby("local_pair_id")
        .agg(
            probe_unit_count=("validation_unit_id", "count"),
            start_condition_count=("start_condition", "nunique"),
            start_conditions=("start_condition", lambda values: ";".join(sorted(map(str, values)))),
        )
        .reset_index()
    )
    rows = pair_candidates.merge(unit_counts, on="local_pair_id", how="left", validate="one_to_one")
    rows["stage2a_pair_contract_status"] = "candidate_pair_in_scoped_pathway_probe_contract"
    rows["planned_route_family_count"] = len(ROUTE_FAMILIES)
    rows["planned_route_row_count"] = rows["probe_unit_count"].astype(int) * len(ROUTE_FAMILIES)
    rows["wall_claim_ready_after_contract"] = False
    rows["required_measurements_before_wall_claim"] = ";".join(REQUIRED_MEASUREMENTS)
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(["local_pair_id"], kind="mergesort").reset_index(drop=True)


def _route_plan_rows(probe_units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit in probe_units.itertuples(index=False):
        for order, family in enumerate(ROUTE_FAMILIES, start=1):
            route_contract_id = (
                f"{unit.validation_unit_id}__{family['planned_route_family']}"
            )
            rows.append(
                {
                    "route_contract_id": route_contract_id,
                    "validation_unit_id": unit.validation_unit_id,
                    "local_pair_id": unit.local_pair_id,
                    "branch": unit.branch,
                    "left_node_id": unit.left_node_id,
                    "right_node_id": unit.right_node_id,
                    "start_condition": unit.start_condition,
                    "route_family_order": order,
                    "planned_route_family": family["planned_route_family"],
                    "route_family_role": family["route_family_role"],
                    "planned_intervention_schedule": family["planned_intervention_schedule"],
                    "expected_endpoint_pattern": family["expected_endpoint_pattern"],
                    "probe_question": family["probe_question"],
                    "original_pair_coassigned_share": unit.original_pair_coassigned_share,
                    "drop_direct_pair_coassigned_share": unit.drop_direct_pair_coassigned_share,
                    "drop_bridge_pair_coassigned_share": unit.drop_bridge_pair_coassigned_share,
                    "drop_direct_and_bridge_pair_coassigned_share": (
                        unit.drop_direct_and_bridge_pair_coassigned_share
                    ),
                    "bridge_release_lift_proxy": unit.bridge_release_lift_proxy,
                    "direct_dependency_proxy": unit.direct_dependency_proxy,
                    "required_measurements": ";".join(REQUIRED_MEASUREMENTS),
                    "route_execution_status": "not_executed_contract_only",
                    "wall_claim_ready_after_contract": False,
                    "quality_cost_claim_allowed": False,
                    "method_claim_allowed": False,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows)


def _control_guard_rows(unit_rows: pd.DataFrame) -> pd.DataFrame:
    controls = unit_rows[~_bool_series(unit_rows["pathway_probe_candidate"])].copy()
    controls["stage2a_contract_role"] = "false_positive_guard_not_route_execution"
    controls["route_execution_permitted_in_scoped_contract"] = False
    controls["route_execution_block_reason"] = controls["pathway_probe_block_reason"]
    controls["wall_claim_ready_after_contract"] = False
    controls["run_status"] = RUN_STATUS
    controls["claim_boundary"] = CLAIM_BOUNDARY
    return controls.sort_values(
        ["limitation_axis", "local_pair_id", "start_condition"], kind="mergesort"
    ).reset_index(drop=True)


def _gate_row(
    gate_id: str, question: str, passed: bool, observed: Any, minimum: Any
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "gate_status": "pass" if bool(passed) else "fail",
        "observed": observed,
        "minimum_or_rule": minimum,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _build_gate_matrix(
    *,
    probe_units: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    control_rows: pd.DataFrame,
    upstream_gates: pd.DataFrame,
) -> pd.DataFrame:
    route_family_counts = _count_dict(route_plan_rows["planned_route_family"])
    rows = [
        _gate_row(
            "G1_upstream_readiness_audit_passes",
            "Did every upstream pathway/wall readiness audit gate pass?",
            bool(upstream_gates["gate_status"].astype(str).eq("pass").all()),
            _count_dict(upstream_gates["gate_status"]),
            "all upstream gates pass",
        ),
        _gate_row(
            "G2_scope_is_two_ready_pairs_only",
            "Is the candidate contract restricted to the two ready pairs?",
            int(len(candidate_pairs)) == 2
            and int(probe_units["local_pair_id"].nunique()) == 2
            and int(len(probe_units)) == 10,
            (
                f"candidate_pairs={len(candidate_pairs)} "
                f"probe_units={len(probe_units)}"
            ),
            "2 candidate pairs and 10 candidate units",
        ),
        _gate_row(
            "G3_all_candidate_starts_preserved",
            "Does each candidate pair retain all five start conditions?",
            bool(
                probe_units.groupby("local_pair_id")["start_condition"].nunique().eq(5).all()
            ),
            json.dumps(
                {
                    str(key): int(value)
                    for key, value in probe_units.groupby("local_pair_id")[
                        "start_condition"
                    ].nunique().items()
                },
                sort_keys=True,
            ),
            "five start conditions for each candidate pair",
        ),
        _gate_row(
            "G4_route_plan_is_tiny_and_predeclared",
            "Are route-plan rows limited to three predeclared families per candidate unit?",
            int(len(route_plan_rows)) == int(len(probe_units) * len(ROUTE_FAMILIES))
            and set(route_plan_rows["planned_route_family"].astype(str))
            == {str(item["planned_route_family"]) for item in ROUTE_FAMILIES},
            f"route_plan_rows={len(route_plan_rows)} route_family_counts={route_family_counts}",
            "30 rows: 10 units times 3 route families",
        ),
        _gate_row(
            "G5_controls_retained_not_executed",
            "Are noncandidate controls retained but excluded from route execution?",
            int(len(control_rows)) == 65
            and not bool(control_rows["route_execution_permitted_in_scoped_contract"].any()),
            f"control_guard_rows={len(control_rows)}",
            "65 false-positive guards, zero control route rows",
        ),
        _gate_row(
            "G6_required_wall_measurements_are_explicit",
            "Are required wall-evidence measurements predeclared for every route row?",
            bool(
                route_plan_rows["required_measurements"].astype(str).eq(
                    ";".join(REQUIRED_MEASUREMENTS)
                ).all()
            ),
            ";".join(REQUIRED_MEASUREMENTS),
            "route trace, objective, endpoint, polish, support fields required",
        ),
        _gate_row(
            "G7_wall_claim_remains_closed",
            "Are wall claims still closed in this contract?",
            not bool(route_plan_rows["wall_claim_ready_after_contract"].any())
            and not bool(candidate_pairs["wall_claim_ready_after_contract"].any()),
            "wall_claim_ready_after_contract=false",
            "design contract only; no executed route evidence",
        ),
        _gate_row(
            "G8_no_new_leiden_execution",
            "Is this a design contract rather than an executed route run?",
            True,
            RUN_STATUS,
            "contract/materialization only",
        ),
        _gate_row(
            "G9_no_method_quality_or_replay_claim",
            "Are method, quality/cost, full replay, and algorithm claims closed?",
            True,
            CLAIM_BOUNDARY,
            "claim boundary explicitly closed",
        ),
    ]
    return pd.DataFrame(rows)


def _contract_status(gate_matrix: pd.DataFrame) -> str:
    if gate_matrix.empty or not bool(gate_matrix["gate_status"].astype(str).eq("pass").all()):
        return "scoped_pathway_probe_contract_gate_failed"
    return "scoped_pathway_probe_contract_ready_wall_claim_closed"


def _build_summary(
    *,
    candidate_pairs: pd.DataFrame,
    probe_units: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    control_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    readiness_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_scoped_pathway_probe_contract_summary.v1",
        "status": _contract_status(gate_matrix),
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "readiness_dir": str(readiness_dir),
        "output_dir": str(output_dir),
        "candidate_pair_count": int(len(candidate_pairs)),
        "probe_unit_count": int(len(probe_units)),
        "route_plan_row_count": int(len(route_plan_rows)),
        "control_guard_row_count": int(len(control_rows)),
        "route_family_counts": _count_dict(route_plan_rows["planned_route_family"]),
        "candidate_pair_ids": list(candidate_pairs["local_pair_id"].astype(str)),
        "control_guard_limitation_axis_counts": _count_dict(control_rows["limitation_axis"]),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": [
            str(row.gate_id)
            for row in gate_matrix.itertuples(index=False)
            if str(row.gate_status) != "pass"
        ],
        "required_measurements": REQUIRED_MEASUREMENTS,
        "interpretation": (
            "The next executable surface, if run, is a tiny Stage 2A pathway "
            "probe: two ready pairs, ten start-conditioned units, and three "
            "predeclared route families per unit. Controls remain false-positive "
            "guards and wall claims remain closed."
        ),
        "recommended_next_gate": (
            "Implement or run only this scoped route-plan contract if proceeding "
            "to execution. Do not broaden to controls or quality/cost/method "
            "claims until route traces and required wall-evidence fields exist."
        ),
        "written_artifacts": [
            CANDIDATE_PAIR_ROWS_CSV,
            PROBE_UNIT_ROWS_CSV,
            ROUTE_PLAN_ROWS_CSV,
            CONTROL_GUARD_ROWS_CSV,
            GATE_MATRIX_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    candidate_pairs: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    control_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Scoped Pathway-Probe Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- candidate_pair_count: {summary['candidate_pair_count']}",
        f"- probe_unit_count: {summary['probe_unit_count']}",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- control_guard_row_count: {summary['control_guard_row_count']}",
        f"- route_family_counts: {summary['route_family_counts']}",
        f"- candidate_pair_ids: {summary['candidate_pair_ids']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- required_measurements: {summary['required_measurements']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Candidate Pairs",
        "",
    ]
    lines.append(
        _markdown_table(
            candidate_pairs,
            [
                "local_pair_id",
                "probe_unit_count",
                "planned_route_row_count",
                "original_pair_coassigned_share_median",
                "drop_bridge_pair_coassigned_share_median",
                "bridge_release_lift_proxy_median",
                "direct_dependency_proxy_median",
            ],
        )
    )
    lines.extend(["", "## Route Families", ""])
    route_family_summary = (
        route_plan_rows.groupby(["planned_route_family", "route_family_role"], sort=True)
        .agg(
            route_plan_row_count=("route_contract_id", "count"),
            candidate_pair_count=("local_pair_id", "nunique"),
            probe_unit_count=("validation_unit_id", "nunique"),
            planned_intervention_schedule=("planned_intervention_schedule", "first"),
            expected_endpoint_pattern=("expected_endpoint_pattern", "first"),
        )
        .reset_index()
    )
    lines.append(
        _markdown_table(
            route_family_summary,
            [
                "planned_route_family",
                "route_family_role",
                "route_plan_row_count",
                "candidate_pair_count",
                "probe_unit_count",
                "planned_intervention_schedule",
                "expected_endpoint_pattern",
            ],
        )
    )
    lines.extend(["", "## Control Guards", ""])
    control_summary = (
        control_rows.groupby(["limitation_axis", "route_execution_block_reason"], sort=True)
        .agg(
            control_guard_row_count=("validation_unit_id", "count"),
            pair_count=("local_pair_id", "nunique"),
        )
        .reset_index()
    )
    lines.append(
        _markdown_table(
            control_summary,
            [
                "limitation_axis",
                "route_execution_block_reason",
                "control_guard_row_count",
                "pair_count",
            ],
        )
    )
    lines.extend(["", "## Gate Matrix", ""])
    lines.append(
        _markdown_table(
            gate_matrix,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
        )
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a design contract for a tiny Stage 2A pathway probe. It does "
            "not execute the route rows. It also does not allow wall claims: the "
            "required route trace, objective, endpoint-assignment, polish, and "
            "support-incompatibility evidence must be produced and audited first.",
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_contract(args: argparse.Namespace) -> dict[str, Any]:
    readiness_dir = Path(args.readiness_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unit_input = _read_csv(readiness_dir / INPUT_UNIT_ROWS_CSV)
    pair_input = _read_csv(readiness_dir / INPUT_PAIR_ROWS_CSV)
    upstream_gates = _read_csv(readiness_dir / INPUT_GATE_MATRIX_CSV)

    probe_units = _candidate_units(unit_input)
    candidate_pairs = _candidate_pairs(pair_input, probe_units)
    route_rows = _route_plan_rows(probe_units)
    control_rows = _control_guard_rows(unit_input)
    gate_matrix = _build_gate_matrix(
        probe_units=probe_units,
        candidate_pairs=candidate_pairs,
        route_plan_rows=route_rows,
        control_rows=control_rows,
        upstream_gates=upstream_gates,
    )
    summary = _build_summary(
        candidate_pairs=candidate_pairs,
        probe_units=probe_units,
        route_plan_rows=route_rows,
        control_rows=control_rows,
        gate_matrix=gate_matrix,
        readiness_dir=readiness_dir,
        output_dir=output_dir,
    )

    _write_csv(candidate_pairs, output_dir / CANDIDATE_PAIR_ROWS_CSV)
    _write_csv(probe_units, output_dir / PROBE_UNIT_ROWS_CSV)
    _write_csv(route_rows, output_dir / ROUTE_PLAN_ROWS_CSV)
    _write_csv(control_rows, output_dir / CONTROL_GUARD_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            _json_safe(
                {
                    "readiness_dir": str(readiness_dir),
                    "output_dir": str(output_dir),
                    "route_families": ROUTE_FAMILIES,
                    "required_measurements": REQUIRED_MEASUREMENTS,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        candidate_pairs=candidate_pairs,
        route_plan_rows=route_rows,
        control_rows=control_rows,
        gate_matrix=gate_matrix,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run_contract(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
