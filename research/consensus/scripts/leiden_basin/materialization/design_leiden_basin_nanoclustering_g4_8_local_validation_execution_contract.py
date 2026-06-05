#!/usr/bin/env python3
"""Freeze the G4.8 local validation execution contract.

This consumes the seed/start validation contract and turns it into explicit
validation units. The primary execution surface is stable-lane only. Conditional
and boundary rows are preserved as secondary and diagnostic lanes, respectively,
so they cannot be mixed into stable validation evidence by accident.

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


DEFAULT_CONTRACT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_seed_start_validation_contract_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_local_validation_execution_contract_gamma1e5_20260604"
)

INPUT_PAIR_ROWS_CSV = "nanoclustering_g4_8_seed_start_validation_contract_pair_rows.csv"
INPUT_START_ROWS_CSV = "nanoclustering_g4_8_seed_start_validation_contract_start_rows.csv"
INPUT_GATE_MATRIX_CSV = "nanoclustering_g4_8_seed_start_validation_contract_gate_matrix.csv"

EXECUTION_PAIR_ROWS_CSV = "nanoclustering_g4_8_local_validation_execution_contract_pair_rows.csv"
EXECUTION_UNIT_ROWS_CSV = "nanoclustering_g4_8_local_validation_execution_contract_unit_rows.csv"
EXECUTION_LANE_SUMMARY_CSV = (
    "nanoclustering_g4_8_local_validation_execution_contract_lane_summary.csv"
)
EXECUTION_BATCH_PLAN_CSV = (
    "nanoclustering_g4_8_local_validation_execution_contract_batch_plan_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_local_validation_execution_contract_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_local_validation_execution_contract_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_local_validation_execution_contract_summary.json"
REPORT_MD = "nanoclustering_g4_8_local_validation_execution_contract_report.md"

START_CONDITIONS = (
    "singleton",
    "pair_together",
    "bridges_to_left",
    "bridges_to_right",
    "all_local_together",
)

RUN_STATUS = "designed_nanoclustering_g4_8_local_validation_execution_contract"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 local validation execution contract design only; reads "
    "the seed/start validation contract and freezes primary stable-lane units, "
    "secondary conditional units, and diagnostic boundary units. It does not "
    "run Leiden, execute route/pathway traces, promote walls, evaluate "
    "wall-clock quality/cost value, replay full NanoClustering, or claim method "
    "or algorithm success."
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


def _execution_phase(execution_lane: str, validation_contract_class: str) -> str:
    if execution_lane == "stable_lane":
        if validation_contract_class == "stable_strict_ready_contract":
            return "primary_stable_ready_validation"
        if validation_contract_class == "stable_target_saturated_noop_contract":
            return "primary_stable_target_saturated_control"
        if validation_contract_class == "stable_latent_release_control_contract":
            return "primary_stable_latent_release_control"
        if validation_contract_class == "stable_no_release_control_contract":
            return "primary_stable_no_release_control"
        if validation_contract_class == "stable_coupled_failure_control_contract":
            return "primary_stable_coupled_failure_control"
        return "primary_stable_validation"
    if execution_lane == "conditional_lane":
        return "secondary_conditional_validation"
    if execution_lane == "boundary_lane":
        return "diagnostic_boundary_validation"
    return "excluded_pending_review"


def _execution_role(execution_lane: str) -> str:
    if execution_lane == "stable_lane":
        return "primary_execution_unit"
    if execution_lane == "conditional_lane":
        return "secondary_allowed_unit"
    if execution_lane == "boundary_lane":
        return "diagnostic_allowed_unit"
    return "excluded_unit"


def _pair_execution_rows(pair_rows: pd.DataFrame, start_rows: pd.DataFrame) -> pd.DataFrame:
    allowed_counts = (
        start_rows[_bool_series(start_rows["start_condition_pass"])]
        .groupby("local_pair_id")
        .size()
        .rename("allowed_execution_unit_count")
        .reset_index()
    )
    rows = pair_rows.merge(allowed_counts, on="local_pair_id", how="left", validate="one_to_one")
    rows["allowed_execution_unit_count"] = (
        rows["allowed_execution_unit_count"].fillna(0).astype(int)
    )
    rows["primary_execution_pair_eligible"] = rows["execution_lane"].eq("stable_lane")
    rows["secondary_execution_pair_eligible"] = rows["execution_lane"].eq("conditional_lane")
    rows["diagnostic_execution_pair_eligible"] = rows["execution_lane"].eq("boundary_lane")
    rows["execution_phase"] = [
        _execution_phase(str(row.execution_lane), str(row.validation_contract_class))
        for row in rows.itertuples(index=False)
    ]
    rows["execution_contract_instruction"] = np.select(
        [
            rows["execution_lane"].eq("stable_lane"),
            rows["execution_lane"].eq("conditional_lane"),
            rows["execution_lane"].eq("boundary_lane"),
        ],
        [
            "include all five start conditions in primary stable validation",
            "include only allowed start conditions as secondary conditional evidence",
            "include only allowed start conditions as diagnostic boundary evidence",
        ],
        default="exclude pending review",
    )
    rows["exact_g4_8f_signature_available"] = False
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["execution_lane", "validation_contract_class", "local_pair_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _validation_unit_rows(pair_rows: pd.DataFrame, start_rows: pd.DataFrame) -> pd.DataFrame:
    pair_cols = [
        "local_pair_id",
        "object_role_universe_id",
        "branch",
        "left_node_id",
        "right_node_id",
        "validation_stratum",
        "validation_family",
        "execution_lane",
        "validation_contract_class",
        "start_stability_class",
        "allowed_start_conditions",
        "blocked_start_conditions",
    ]
    pairs = pair_rows[[col for col in pair_cols if col in pair_rows.columns]].copy()
    starts = start_rows.copy()
    rows = starts.merge(pairs, on="local_pair_id", how="left", validate="many_to_one")
    rows = rows[_bool_series(rows["start_condition_pass"])].copy()
    rows["validation_unit_id"] = (
        rows["local_pair_id"].astype(str) + "__" + rows["start_condition"].astype(str)
    )
    rows["execution_phase"] = [
        _execution_phase(str(row.execution_lane), str(row.validation_contract_class))
        for row in rows.itertuples(index=False)
    ]
    rows["execution_unit_role"] = rows["execution_lane"].astype(str).map(_execution_role)
    rows["include_in_primary_execution"] = rows["execution_lane"].eq("stable_lane")
    rows["include_in_secondary_execution"] = rows["execution_lane"].eq("conditional_lane")
    rows["include_as_diagnostic_control"] = rows["execution_lane"].eq("boundary_lane")
    rows["blocked_from_primary_reason"] = np.select(
        [
            rows["execution_lane"].eq("stable_lane"),
            rows["execution_lane"].eq("conditional_lane"),
            rows["execution_lane"].eq("boundary_lane"),
        ],
        [
            "",
            "conditional lane; held-out passes but not all start conditions pass",
            "boundary lane; held-out stratum shifted",
        ],
        default="not in executable lane",
    )
    rows["no_new_leiden_execution"] = True
    rows["exact_g4_8f_signature_available"] = False
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["execution_lane", "validation_contract_class", "local_pair_id", "start_condition"],
        kind="mergesort",
    ).reset_index(drop=True)


def _summary_table(pair_rows: pd.DataFrame, unit_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lane, group in pair_rows.groupby("execution_lane", sort=True):
        unit_group = unit_rows[unit_rows["execution_lane"].astype(str).eq(str(lane))]
        data: dict[str, Any] = {
            "execution_lane": str(lane),
            "pair_count": int(len(group)),
            "validation_unit_count": int(len(unit_group)),
            "primary_unit_count": int(unit_group["include_in_primary_execution"].sum()),
            "secondary_unit_count": int(unit_group["include_in_secondary_execution"].sum()),
            "diagnostic_unit_count": int(unit_group["include_as_diagnostic_control"].sum()),
            "validation_stratum_counts": json.dumps(
                _count_dict(group["validation_stratum"]),
                sort_keys=True,
            ),
            "validation_contract_class_counts": json.dumps(
                _count_dict(group["validation_contract_class"]),
                sort_keys=True,
            ),
            "start_condition_counts": json.dumps(
                _count_dict(unit_group["start_condition"]) if not unit_group.empty else {},
                sort_keys=True,
            ),
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        if "allowed_execution_unit_count" in group.columns:
            data.update(
                _prefix_stats(
                    "allowed_execution_unit_count", group["allowed_execution_unit_count"]
                )
            )
        rows.append(data)
    return pd.DataFrame(rows)


def _batch_plan_rows(pair_rows: pd.DataFrame, unit_rows: pd.DataFrame) -> pd.DataFrame:
    batch_specs = [
        (
            "B1_primary_stable_ready",
            "stable_lane",
            "stable_strict_ready_contract",
            "primary",
            "Ready candidates in the stable lane; first local validation target.",
        ),
        (
            "B2_primary_stable_target_saturated_control",
            "stable_lane",
            "stable_target_saturated_noop_contract",
            "primary_control",
            "Stable target-saturated no-op controls.",
        ),
        (
            "B3_primary_stable_latent_release_control",
            "stable_lane",
            "stable_latent_release_control_contract",
            "primary_control",
            "Stable latent-release controls without original source.",
        ),
        (
            "B4_primary_stable_no_release_control",
            "stable_lane",
            "stable_no_release_control_contract",
            "primary_control",
            "Stable no-release hard negative controls.",
        ),
        (
            "B5_primary_stable_coupled_failure_control",
            "stable_lane",
            "stable_coupled_failure_control_contract",
            "primary_control",
            "Stable coupled direct-bridge failure controls.",
        ),
        (
            "B6_secondary_conditional_allowed_starts",
            "conditional_lane",
            None,
            "secondary",
            "Conditional rows; use only listed allowed start conditions.",
        ),
        (
            "B7_diagnostic_boundary_allowed_starts",
            "boundary_lane",
            None,
            "diagnostic",
            "Boundary rows; diagnostic controls only, not stable evidence.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for order, (batch_id, lane, contract_class, batch_role, instruction) in enumerate(
        batch_specs, start=1
    ):
        pair_mask = pair_rows["execution_lane"].astype(str).eq(lane)
        unit_mask = unit_rows["execution_lane"].astype(str).eq(lane)
        if contract_class is not None:
            pair_mask &= pair_rows["validation_contract_class"].astype(str).eq(contract_class)
            unit_mask &= unit_rows["validation_contract_class"].astype(str).eq(contract_class)
        pair_group = pair_rows[pair_mask]
        unit_group = unit_rows[unit_mask]
        rows.append(
            {
                "batch_order": order,
                "batch_id": batch_id,
                "execution_lane": lane,
                "validation_contract_class": contract_class or "all_lane_classes",
                "batch_role": batch_role,
                "pair_count": int(len(pair_group)),
                "validation_unit_count": int(len(unit_group)),
                "start_condition_counts": json.dumps(
                    _count_dict(unit_group["start_condition"]) if not unit_group.empty else {},
                    sort_keys=True,
                ),
                "local_pair_ids": ";".join(pair_group["local_pair_id"].astype(str)),
                "instruction": instruction,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


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
    pair_rows: pd.DataFrame,
    unit_rows: pd.DataFrame,
    batch_rows: pd.DataFrame,
    upstream_gates: pd.DataFrame,
) -> pd.DataFrame:
    stable_pairs = pair_rows[pair_rows["execution_lane"].eq("stable_lane")]
    conditional_pairs = pair_rows[pair_rows["execution_lane"].eq("conditional_lane")]
    boundary_pairs = pair_rows[pair_rows["execution_lane"].eq("boundary_lane")]
    primary_units = unit_rows[unit_rows["include_in_primary_execution"]]
    secondary_units = unit_rows[unit_rows["include_in_secondary_execution"]]
    diagnostic_units = unit_rows[unit_rows["include_as_diagnostic_control"]]
    rows = [
        _gate_row(
            "G1_upstream_seed_start_contract_passes",
            "Did every upstream seed/start contract gate pass?",
            bool(upstream_gates["gate_status"].astype(str).eq("pass").all()),
            _count_dict(upstream_gates["gate_status"]),
            "all upstream gates pass",
        ),
        _gate_row(
            "G2_primary_surface_is_stable_lane_only",
            "Is primary validation restricted to stable-lane pairs?",
            int(len(stable_pairs)) == 15
            and bool(primary_units["execution_lane"].astype(str).eq("stable_lane").all()),
            f"stable_pairs={len(stable_pairs)} primary_units={len(primary_units)}",
            "15 stable pairs; primary units stable-lane only",
        ),
        _gate_row(
            "G3_primary_starts_are_complete",
            "Does every stable pair carry all five start conditions?",
            int(len(primary_units)) == int(len(stable_pairs) * len(START_CONDITIONS))
            and int(
                primary_units.groupby("local_pair_id")["start_condition"].nunique().min()
            )
            == len(START_CONDITIONS),
            f"primary_units={len(primary_units)} min_primary_starts={int(primary_units.groupby('local_pair_id')['start_condition'].nunique().min())}",
            "5 start conditions for each stable pair",
        ),
        _gate_row(
            "G4_primary_has_ready_plus_controls",
            "Does primary validation include stable ready rows and all control roles?",
            bool(
                {
                    "stable_strict_ready_contract",
                    "stable_target_saturated_noop_contract",
                    "stable_latent_release_control_contract",
                    "stable_no_release_control_contract",
                    "stable_coupled_failure_control_contract",
                }.issubset(set(stable_pairs["validation_contract_class"].astype(str)))
            ),
            json.dumps(_count_dict(stable_pairs["validation_contract_class"]), sort_keys=True),
            "stable strict-ready plus target/no-release/latent/coupled controls",
        ),
        _gate_row(
            "G5_conditional_lane_is_secondary_only",
            "Are conditional rows excluded from primary and retained as secondary allowed starts?",
            int(len(conditional_pairs)) == 5
            and int(len(secondary_units)) == 16
            and not bool(secondary_units["include_in_primary_execution"].any()),
            f"conditional_pairs={len(conditional_pairs)} secondary_units={len(secondary_units)}",
            "5 conditional pairs and 16 allowed secondary units",
        ),
        _gate_row(
            "G6_boundary_lane_is_diagnostic_only",
            "Are boundary rows excluded from primary and retained only as diagnostic allowed starts?",
            int(len(boundary_pairs)) == 3
            and int(len(diagnostic_units)) == 10
            and not bool(diagnostic_units["include_in_primary_execution"].any()),
            f"boundary_pairs={len(boundary_pairs)} diagnostic_units={len(diagnostic_units)}",
            "3 boundary pairs and 10 allowed diagnostic units",
        ),
        _gate_row(
            "G7_blocked_start_conditions_excluded",
            "Are blocked start conditions excluded from validation unit rows?",
            bool(unit_rows["start_condition_contract_role"].astype(str).eq("allowed_start_condition").all()),
            json.dumps(_count_dict(unit_rows["start_condition_contract_role"]), sort_keys=True),
            "validation units contain allowed starts only",
        ),
        _gate_row(
            "G8_batch_plan_covers_all_units_once",
            "Does the batch plan cover every validation unit exactly once by lane/class?",
            int(batch_rows["validation_unit_count"].sum()) == int(len(unit_rows)),
            f"batch_units={int(batch_rows['validation_unit_count'].sum())} unit_rows={len(unit_rows)}",
            "batch unit total equals validation unit rows",
        ),
        _gate_row(
            "G9_exact_signature_gap_closed",
            "Is the exact G4.8F source-signature gap kept closed?",
            not bool(pair_rows["exact_g4_8f_signature_available"].fillna(False).astype(bool).any())
            and not bool(
                unit_rows["exact_g4_8f_signature_available"].fillna(False).astype(bool).any()
            ),
            "exact_g4_8f_signature_available=false",
            "proxy signatures only",
        ),
        _gate_row(
            "G10_no_new_leiden_execution",
            "Is this a contract over existing seed/start rows rather than a new run?",
            True,
            RUN_STATUS,
            "design/materialization only",
        ),
        _gate_row(
            "G11_no_method_or_wall_claim",
            "Are replay, wall/pathway, quality/cost, and method claims closed?",
            True,
            CLAIM_BOUNDARY,
            "claim boundary explicitly closed",
        ),
    ]
    return pd.DataFrame(rows)


def _contract_status(gate_matrix: pd.DataFrame) -> str:
    if gate_matrix.empty or not bool(gate_matrix["gate_status"].astype(str).eq("pass").all()):
        return "local_validation_execution_contract_gate_failed"
    return "local_validation_execution_contract_ready_stable_primary"


def _build_summary(
    *,
    pair_rows: pd.DataFrame,
    unit_rows: pd.DataFrame,
    lane_summary: pd.DataFrame,
    batch_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    contract_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    primary_units = unit_rows[unit_rows["include_in_primary_execution"]]
    secondary_units = unit_rows[unit_rows["include_in_secondary_execution"]]
    diagnostic_units = unit_rows[unit_rows["include_as_diagnostic_control"]]
    return {
        "schema": "nanoclustering_g4_8_local_validation_execution_contract_summary.v1",
        "status": _contract_status(gate_matrix),
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "contract_dir": str(contract_dir),
        "output_dir": str(output_dir),
        "pair_count": int(len(pair_rows)),
        "validation_unit_count": int(len(unit_rows)),
        "primary_stable_pair_count": int(pair_rows["execution_lane"].eq("stable_lane").sum()),
        "conditional_pair_count": int(pair_rows["execution_lane"].eq("conditional_lane").sum()),
        "boundary_pair_count": int(pair_rows["execution_lane"].eq("boundary_lane").sum()),
        "primary_validation_unit_count": int(len(primary_units)),
        "secondary_validation_unit_count": int(len(secondary_units)),
        "diagnostic_validation_unit_count": int(len(diagnostic_units)),
        "execution_lane_counts": _count_dict(pair_rows["execution_lane"]),
        "validation_contract_class_counts": _count_dict(pair_rows["validation_contract_class"]),
        "validation_unit_lane_counts": _count_dict(unit_rows["execution_lane"]),
        "validation_unit_phase_counts": _count_dict(unit_rows["execution_phase"]),
        "lane_summary_rows": int(len(lane_summary)),
        "batch_plan_rows": int(len(batch_rows)),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": [
            str(row.gate_id)
            for row in gate_matrix.itertuples(index=False)
            if str(row.gate_status) != "pass"
        ],
        "exact_g4_8f_signature_available": False,
        "recommended_next_gate": (
            "Execute or simulate the primary stable-lane validation units first. "
            "Report conditional allowed starts and boundary diagnostics in separate "
            "sections; do not mix them into stable validation evidence before any "
            "route/pathway, quality/cost, full NanoClustering replay, or method claim."
        ),
        "written_artifacts": [
            EXECUTION_PAIR_ROWS_CSV,
            EXECUTION_UNIT_ROWS_CSV,
            EXECUTION_LANE_SUMMARY_CSV,
            EXECUTION_BATCH_PLAN_CSV,
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
    lane_summary: pd.DataFrame,
    batch_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    pair_rows: pd.DataFrame,
) -> None:
    stable_pairs = pair_rows[pair_rows["execution_lane"].eq("stable_lane")]
    conditional_pairs = pair_rows[pair_rows["execution_lane"].eq("conditional_lane")]
    boundary_pairs = pair_rows[pair_rows["execution_lane"].eq("boundary_lane")]
    lines = [
        "# NanoClustering G4.8 Local Validation Execution Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_count: {summary['pair_count']}",
        f"- validation_unit_count: {summary['validation_unit_count']}",
        f"- primary_stable_pair_count: {summary['primary_stable_pair_count']}",
        f"- primary_validation_unit_count: {summary['primary_validation_unit_count']}",
        f"- secondary_validation_unit_count: {summary['secondary_validation_unit_count']}",
        f"- diagnostic_validation_unit_count: {summary['diagnostic_validation_unit_count']}",
        f"- execution_lane_counts: {summary['execution_lane_counts']}",
        f"- validation_contract_class_counts: {summary['validation_contract_class_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- exact_g4_8f_signature_available: {summary['exact_g4_8f_signature_available']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Lane Summary",
        "",
    ]
    lines.append(
        _markdown_table(
            lane_summary,
            [
                "execution_lane",
                "pair_count",
                "validation_unit_count",
                "primary_unit_count",
                "secondary_unit_count",
                "diagnostic_unit_count",
                "validation_contract_class_counts",
                "start_condition_counts",
            ],
        )
    )
    lines.extend(["", "## Batch Plan", ""])
    lines.append(
        _markdown_table(
            batch_rows,
            [
                "batch_order",
                "batch_id",
                "batch_role",
                "execution_lane",
                "validation_contract_class",
                "pair_count",
                "validation_unit_count",
                "start_condition_counts",
            ],
        )
    )
    lines.extend(["", "## Primary Stable Pairs", ""])
    lines.append(
        _markdown_table(
            stable_pairs,
            [
                "local_pair_id",
                "validation_stratum",
                "validation_contract_class",
                "allowed_execution_unit_count",
                "execution_contract_instruction",
            ],
        )
    )
    lines.extend(["", "## Secondary Conditional Pairs", ""])
    lines.append(
        _markdown_table(
            conditional_pairs,
            [
                "local_pair_id",
                "validation_stratum",
                "validation_contract_class",
                "allowed_start_conditions",
                "blocked_start_conditions",
                "allowed_execution_unit_count",
            ],
        )
    )
    lines.extend(["", "## Diagnostic Boundary Pairs", ""])
    lines.append(
        _markdown_table(
            boundary_pairs,
            [
                "local_pair_id",
                "validation_stratum",
                "validation_contract_class",
                "heldout_seed_split_macro_role",
                "allowed_start_conditions",
                "blocked_start_conditions",
                "allowed_execution_unit_count",
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
            "This contract is the execution boundary for the next local validation "
            "step. Primary validation is stable-lane only. Conditional rows can "
            "be evaluated only as secondary allowed-start evidence. Boundary rows "
            "are diagnostic controls. This artifact does not run Leiden and does "
            "not open route/pathway, wall, quality/cost, full NanoClustering "
            "replay, or method claims.",
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract_dir = Path(args.contract_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_pairs = _read_csv(contract_dir / INPUT_PAIR_ROWS_CSV)
    input_starts = _read_csv(contract_dir / INPUT_START_ROWS_CSV)
    input_gates = _read_csv(contract_dir / INPUT_GATE_MATRIX_CSV)

    pair_rows = _pair_execution_rows(input_pairs, input_starts)
    unit_rows = _validation_unit_rows(pair_rows, input_starts)
    lane_summary = _summary_table(pair_rows, unit_rows)
    batch_rows = _batch_plan_rows(pair_rows, unit_rows)
    gate_matrix = _build_gate_matrix(
        pair_rows=pair_rows,
        unit_rows=unit_rows,
        batch_rows=batch_rows,
        upstream_gates=input_gates,
    )
    summary = _build_summary(
        pair_rows=pair_rows,
        unit_rows=unit_rows,
        lane_summary=lane_summary,
        batch_rows=batch_rows,
        gate_matrix=gate_matrix,
        contract_dir=contract_dir,
        output_dir=output_dir,
    )

    _write_csv(pair_rows, output_dir / EXECUTION_PAIR_ROWS_CSV)
    _write_csv(unit_rows, output_dir / EXECUTION_UNIT_ROWS_CSV)
    _write_csv(lane_summary, output_dir / EXECUTION_LANE_SUMMARY_CSV)
    _write_csv(batch_rows, output_dir / EXECUTION_BATCH_PLAN_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            _json_safe(
                {
                    "contract_dir": str(contract_dir),
                    "output_dir": str(output_dir),
                    "start_conditions": START_CONDITIONS,
                    "primary_rule": "stable lane only; all five start conditions",
                    "secondary_rule": "conditional lane allowed starts only",
                    "diagnostic_rule": "boundary lane allowed starts only",
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
        lane_summary=lane_summary,
        batch_rows=batch_rows,
        gate_matrix=gate_matrix,
        pair_rows=pair_rows,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run_contract(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
