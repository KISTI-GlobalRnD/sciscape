#!/usr/bin/env python3
"""Freeze a seed/start-stratified contract from the G4.8 local readout.

This consumes the read-only NanoClustering G4.8 local validation readout and
turns its held-out fragility into an explicit validation contract. It does not
run Leiden. It separates stable rows, start-conditional rows, and boundary rows
before any route/pathway, quality/cost, full NanoClustering replay, or method
claim.
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


DEFAULT_READOUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_local_validation_readout_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_seed_start_validation_contract_gamma1e5_20260604"
)

READOUT_PAIR_ROWS_CSV = "nanoclustering_g4_8_local_validation_readout_pair_rows.csv"
READOUT_START_ROWS_CSV = "nanoclustering_g4_8_local_validation_readout_start_condition_rows.csv"
READOUT_GATE_MATRIX_CSV = "nanoclustering_g4_8_local_validation_readout_gate_matrix.csv"

CONTRACT_PAIR_ROWS_CSV = "nanoclustering_g4_8_seed_start_validation_contract_pair_rows.csv"
CONTRACT_START_ROWS_CSV = "nanoclustering_g4_8_seed_start_validation_contract_start_rows.csv"
CONTRACT_STRATUM_SUMMARY_CSV = (
    "nanoclustering_g4_8_seed_start_validation_contract_stratum_summary.csv"
)
CONTRACT_CLASS_SUMMARY_CSV = (
    "nanoclustering_g4_8_seed_start_validation_contract_class_summary.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_seed_start_validation_contract_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_seed_start_validation_contract_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_seed_start_validation_contract_summary.json"
REPORT_MD = "nanoclustering_g4_8_seed_start_validation_contract_report.md"

START_CONDITIONS = (
    "singleton",
    "pair_together",
    "bridges_to_left",
    "bridges_to_right",
    "all_local_together",
)

RUN_STATUS = "designed_nanoclustering_g4_8_seed_start_validation_contract"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 seed/start validation contract design only; reads the "
    "local validation readout and stratifies rows into stable, conditional, and "
    "boundary lanes. It does not run Leiden, execute route/pathway traces, "
    "promote walls, evaluate wall-clock quality/cost value, replay full "
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


def _start_condition_contract(start_rows: pd.DataFrame) -> pd.DataFrame:
    rows = start_rows.copy()
    rows["start_condition_pass"] = _bool_series(rows["start_condition_expected_validation_pass"])
    rows["start_condition_contract_role"] = np.where(
        rows["start_condition_pass"],
        "allowed_start_condition",
        "blocked_start_condition",
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(["local_pair_id", "start_condition"], kind="mergesort").reset_index(
        drop=True
    )


def _start_lists(start_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for local_pair_id, group in start_rows.groupby("local_pair_id", sort=False):
        passed = group[_bool_series(group["start_condition_expected_validation_pass"])]
        failed = group[~_bool_series(group["start_condition_expected_validation_pass"])]
        rows.append(
            {
                "local_pair_id": str(local_pair_id),
                "start_pass_count": int(len(passed)),
                "start_fail_count": int(len(failed)),
                "allowed_start_conditions": ";".join(
                    str(value) for value in sorted(passed["start_condition"].astype(str))
                ),
                "blocked_start_conditions": ";".join(
                    str(value) for value in sorted(failed["start_condition"].astype(str))
                ),
                "start_macro_role_counts": json.dumps(
                    _count_dict(group["start_condition_macro_role"]),
                    sort_keys=True,
                ),
                "start_source_condition_counts": json.dumps(
                    _count_dict(group["start_condition_source_condition"]),
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def _start_stability_class(start_pass_count: int) -> str:
    count = int(start_pass_count)
    if count == len(START_CONDITIONS):
        return "start_invariant"
    if count >= 3:
        return "start_majority_conditional"
    if count >= 1:
        return "start_rare_conditional"
    return "start_absent"


def _execution_lane(*, heldout_pass: bool, start_pass_count: int) -> str:
    if not bool(heldout_pass):
        return "boundary_lane"
    if int(start_pass_count) == len(START_CONDITIONS):
        return "stable_lane"
    if int(start_pass_count) > 0:
        return "conditional_lane"
    return "blocked_lane"


def _contract_class(row: pd.Series) -> tuple[str, str]:
    stratum = str(row["validation_stratum"])
    heldout_pass = bool(row["heldout_seed_split_expected_validation_pass"])
    heldout_macro = str(row["heldout_seed_split_macro_role"])
    start_pass_count = int(row["start_pass_count"])
    lane = _execution_lane(heldout_pass=heldout_pass, start_pass_count=start_pass_count)

    if lane == "boundary_lane":
        if stratum == "rare_ready" and heldout_macro == "N_like":
            return (
                "rare_ready_latent_release_boundary_contract",
                "boundary row; use as seed/start fragility evidence, not stable ready",
            )
        if stratum == "strict_ready" and heldout_macro == "T_like":
            return (
                "strict_ready_target_saturation_boundary_contract",
                "boundary row; strict-ready surface saturates under held-out seeds",
            )
        if stratum == "target_saturated_no_handle" and heldout_macro == "R_candidate":
            return (
                "target_saturated_threshold_ready_boundary_contract",
                "boundary row; target-saturated surface touches strict-ready threshold",
            )
        return (
            "unclassified_boundary_contract",
            "boundary row; inspect before any validation execution",
        )

    if stratum == "strict_ready":
        if lane == "stable_lane":
            return (
                "stable_strict_ready_contract",
                "stable ready candidate; eligible for stable-lane local validation",
            )
        return (
            "conditional_strict_ready_contract",
            "ready candidate; use only under allowed start conditions",
        )
    if stratum == "rare_ready":
        return (
            "rare_ready_seed_start_conditional_contract",
            "rare ready candidate; preserve as conditional evidence only",
        )
    if stratum == "target_saturated_no_handle":
        if lane == "stable_lane":
            return (
                "stable_target_saturated_noop_contract",
                "stable no-op target-saturated control",
            )
        return (
            "conditional_target_saturated_noop_contract",
            "target-saturated control with start-condition caveat",
        )
    if stratum == "latent_release_no_source_control":
        return (
            "stable_latent_release_control_contract",
            "stable latent-release control; release exists without original source",
        )
    if stratum == "no_release_control":
        return (
            "stable_no_release_control_contract",
            "stable hard negative no-release control",
        )
    if stratum == "coupled_direct_bridge_failure_control":
        return (
            "stable_coupled_failure_control_contract",
            "stable coupled direct-bridge failure control",
        )
    return (
        "unclassified_contract",
        "unclassified row; inspect before use",
    )


def _contract_pair_rows(pair_rows: pd.DataFrame, start_rows: pd.DataFrame) -> pd.DataFrame:
    starts = _start_lists(start_rows)
    rows = pair_rows.merge(starts, on="local_pair_id", how="left", validate="one_to_one")
    rows["start_stability_class"] = rows["start_pass_count"].astype(int).map(
        _start_stability_class
    )
    rows["execution_lane"] = [
        _execution_lane(
            heldout_pass=bool(row.heldout_seed_split_expected_validation_pass),
            start_pass_count=int(row.start_pass_count),
        )
        for row in rows.itertuples(index=False)
    ]
    classes = rows.apply(_contract_class, axis=1)
    rows["validation_contract_class"] = [item[0] for item in classes]
    rows["validation_contract_rationale"] = [item[1] for item in classes]
    rows["stable_contract_eligible"] = rows["execution_lane"].eq("stable_lane")
    rows["conditional_contract_eligible"] = rows["execution_lane"].eq("conditional_lane")
    rows["boundary_contract_eligible"] = rows["execution_lane"].eq("boundary_lane")
    rows["next_execution_instruction"] = np.select(
        [
            rows["execution_lane"].eq("stable_lane"),
            rows["execution_lane"].eq("conditional_lane"),
            rows["execution_lane"].eq("boundary_lane"),
        ],
        [
            "eligible for stable-lane validation contract",
            "eligible only with listed allowed start conditions",
            "boundary lane only; do not count as stable validation evidence",
        ],
        default="blocked pending review",
    )
    rows["exact_g4_8f_signature_available"] = False
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["execution_lane", "validation_contract_class", "local_pair_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _summary_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        data: dict[str, Any] = dict(zip(group_cols, key, strict=True))
        data.update(
            {
                "pair_count": int(len(group)),
                "stable_lane_count": int(group["execution_lane"].eq("stable_lane").sum()),
                "conditional_lane_count": int(
                    group["execution_lane"].eq("conditional_lane").sum()
                ),
                "boundary_lane_count": int(group["execution_lane"].eq("boundary_lane").sum()),
                "heldout_expected_pass_count": int(
                    _bool_series(group["heldout_seed_split_expected_validation_pass"]).sum()
                ),
                "start_invariant_count": int(
                    group["start_stability_class"].eq("start_invariant").sum()
                ),
                "start_majority_conditional_count": int(
                    group["start_stability_class"].eq("start_majority_conditional").sum()
                ),
                "start_rare_conditional_count": int(
                    group["start_stability_class"].eq("start_rare_conditional").sum()
                ),
                "contract_class_counts": json.dumps(
                    _count_dict(group["validation_contract_class"]),
                    sort_keys=True,
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        for col in [
            "heldout_original_pair_coassigned_share",
            "heldout_drop_bridge_pair_coassigned_share",
            "heldout_original_source_endpoint_signature_proxy_count",
            "heldout_original_coassigned_signature_count",
            "start_pass_count",
        ]:
            if col in group.columns:
                data.update(_prefix_stats(col, group[col]))
        rows.append(data)
    return pd.DataFrame(rows)


def _gate_row(gate_id: str, question: str, passed: bool, observed: Any, minimum: Any) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "gate_status": "pass" if bool(passed) else "fail",
        "observed": observed,
        "minimum_or_rule": minimum,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _build_gate_matrix(pair_rows: pd.DataFrame, start_rows: pd.DataFrame, readout_gates: pd.DataFrame) -> pd.DataFrame:
    lane_counts = _count_dict(pair_rows["execution_lane"])
    boundary = pair_rows[pair_rows["execution_lane"].eq("boundary_lane")]
    stable = pair_rows[pair_rows["execution_lane"].eq("stable_lane")]
    conditional = pair_rows[pair_rows["execution_lane"].eq("conditional_lane")]
    readout_material_gates = readout_gates[
        ~readout_gates["gate_id"].astype(str).eq("G5_heldout_stratum_stability")
    ]
    rows = [
        _gate_row(
            "G1_contract_preserves_readout_rows",
            "Does the contract preserve all 23 readout pairs?",
            int(pair_rows["local_pair_id"].nunique()) == 23,
            f"pair_count={pair_rows['local_pair_id'].nunique()}",
            "all 23 readout pairs",
        ),
        _gate_row(
            "G2_start_contract_complete",
            "Does every pair have all five start-condition contracts?",
            int(start_rows.groupby("local_pair_id")["start_condition"].nunique().min()) == len(START_CONDITIONS),
            f"min_start_conditions={int(start_rows.groupby('local_pair_id')['start_condition'].nunique().min())}",
            "5 start conditions per pair",
        ),
        _gate_row(
            "G3_readout_material_gates_preserved",
            "Did the upstream readout material gates pass after excluding expected held-out fragility?",
            bool(readout_material_gates["gate_status"].astype(str).eq("pass").all()),
            _count_dict(readout_material_gates["gate_status"]),
            "all non-stability readout gates pass",
        ),
        _gate_row(
            "G4_stable_lane_has_ready_and_controls",
            "Does the stable lane include ready and control rows?",
            bool(stable["validation_stratum"].isin(["strict_ready"]).any())
            and bool(stable["validation_family"].isin(["target_saturated", "nonready_control", "failure_control"]).any()),
            json.dumps(_count_dict(stable["validation_stratum"]), sort_keys=True),
            "stable strict-ready plus stable controls",
        ),
        _gate_row(
            "G5_conditional_lane_isolated",
            "Are start-dependent rows isolated into the conditional lane?",
            int(len(conditional)) == 5,
            f"conditional_lane_count={len(conditional)}",
            "5 heldout-pass but start-dependent rows isolated",
        ),
        _gate_row(
            "G6_boundary_lane_isolated",
            "Are the three held-out fragile rows isolated into the boundary lane?",
            int(len(boundary)) == 3
            and bool(boundary["heldout_seed_split_expected_validation_pass"].eq(False).all()),
            f"boundary_lane_count={len(boundary)}",
            "3 held-out fragile rows isolated",
        ),
        _gate_row(
            "G7_contract_classes_cover_all_expected_roles",
            "Do contract classes cover stable, conditional, and boundary roles?",
            {"stable_lane", "conditional_lane", "boundary_lane"}.issubset(
                set(pair_rows["execution_lane"].astype(str))
            ),
            json.dumps(lane_counts, sort_keys=True),
            "stable, conditional, and boundary lanes present",
        ),
        _gate_row(
            "G8_exact_signature_gap_closed",
            "Is the exact G4.8F source-signature gap kept closed?",
            not bool(pair_rows["exact_g4_8f_signature_available"].fillna(False).astype(bool).any()),
            "exact_g4_8f_signature_available=false",
            "proxy signatures only",
        ),
        _gate_row(
            "G9_no_new_leiden_execution",
            "Is this a contract over existing readout rows rather than a new run?",
            True,
            RUN_STATUS,
            "design/materialization only",
        ),
        _gate_row(
            "G10_no_method_or_wall_claim",
            "Are replay, wall/pathway, quality/cost, and method claims closed?",
            True,
            CLAIM_BOUNDARY,
            "claim boundary explicitly closed",
        ),
    ]
    return pd.DataFrame(rows)


def _contract_status(gate_matrix: pd.DataFrame) -> str:
    if gate_matrix.empty or not bool(gate_matrix["gate_status"].astype(str).eq("pass").all()):
        return "seed_start_validation_contract_gate_failed"
    return "seed_start_validation_contract_ready_with_boundary_lanes"


def _build_summary(
    *,
    pair_rows: pd.DataFrame,
    start_rows: pd.DataFrame,
    stratum_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    readout_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    lane_counts = _count_dict(pair_rows["execution_lane"])
    return {
        "schema": "nanoclustering_g4_8_seed_start_validation_contract_summary.v1",
        "status": _contract_status(gate_matrix),
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "readout_dir": str(readout_dir),
        "output_dir": str(output_dir),
        "pair_count": int(len(pair_rows)),
        "start_contract_rows": int(len(start_rows)),
        "stratum_summary_rows": int(len(stratum_summary)),
        "class_summary_rows": int(len(class_summary)),
        "execution_lane_counts": lane_counts,
        "validation_contract_class_counts": _count_dict(pair_rows["validation_contract_class"]),
        "stable_lane_count": int(pair_rows["execution_lane"].eq("stable_lane").sum()),
        "conditional_lane_count": int(pair_rows["execution_lane"].eq("conditional_lane").sum()),
        "boundary_lane_count": int(pair_rows["execution_lane"].eq("boundary_lane").sum()),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": [
            str(row.gate_id)
            for row in gate_matrix.itertuples(index=False)
            if str(row.gate_status) != "pass"
        ],
        "exact_g4_8f_signature_available": False,
        "recommended_next_gate": (
            "Use stable lanes for the next local validation contract, treat "
            "conditional lanes only under allowed start conditions, and keep "
            "boundary lanes as diagnostic controls before any route/pathway, "
            "quality/cost, full NanoClustering replay, or method claim."
        ),
        "written_artifacts": [
            CONTRACT_PAIR_ROWS_CSV,
            CONTRACT_START_ROWS_CSV,
            CONTRACT_STRATUM_SUMMARY_CSV,
            CONTRACT_CLASS_SUMMARY_CSV,
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
    stratum_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    pair_rows: pd.DataFrame,
) -> None:
    boundary = pair_rows[pair_rows["execution_lane"].eq("boundary_lane")]
    lines = [
        "# NanoClustering G4.8 Seed/Start Validation Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_count: {summary['pair_count']}",
        f"- execution_lane_counts: {summary['execution_lane_counts']}",
        f"- validation_contract_class_counts: {summary['validation_contract_class_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- exact_g4_8f_signature_available: {summary['exact_g4_8f_signature_available']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Stratum Summary",
        "",
    ]
    lines.append(
        _markdown_table(
            stratum_summary,
            [
                "validation_stratum",
                "pair_count",
                "stable_lane_count",
                "conditional_lane_count",
                "boundary_lane_count",
                "start_invariant_count",
                "contract_class_counts",
            ],
        )
    )
    lines.extend(["", "## Contract Class Summary", ""])
    lines.append(
        _markdown_table(
            class_summary,
            [
                "validation_contract_class",
                "pair_count",
                "stable_lane_count",
                "conditional_lane_count",
                "boundary_lane_count",
                "start_pass_count_median",
            ],
        )
    )
    lines.extend(["", "## Boundary Rows", ""])
    if boundary.empty:
        lines.append("No boundary rows.")
    else:
        lines.append(
            _markdown_table(
                boundary,
                [
                    "local_pair_id",
                    "validation_stratum",
                    "validation_contract_class",
                    "heldout_seed_split_macro_role",
                    "heldout_seed_split_source_condition",
                    "start_pass_count",
                    "allowed_start_conditions",
                    "blocked_start_conditions",
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
            "This contract freezes how to use the local validation readout. Stable "
            "rows can feed the next local validation contract, conditional rows "
            "must be restricted to their allowed start conditions, and boundary "
            "rows remain diagnostic controls. It does not run Leiden and does not "
            "open route/pathway, wall, quality/cost, full NanoClustering replay, "
            "or method claims.",
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_contract(args: argparse.Namespace) -> dict[str, Any]:
    readout_dir = Path(args.readout_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_readout = _read_csv(readout_dir / READOUT_PAIR_ROWS_CSV)
    start_readout = _read_csv(readout_dir / READOUT_START_ROWS_CSV)
    readout_gates = _read_csv(readout_dir / READOUT_GATE_MATRIX_CSV)

    start_rows = _start_condition_contract(start_readout)
    pair_rows = _contract_pair_rows(pair_readout, start_rows)
    stratum_summary = _summary_table(pair_rows, ["validation_stratum"])
    class_summary = _summary_table(pair_rows, ["validation_contract_class"])
    gate_matrix = _build_gate_matrix(pair_rows, start_rows, readout_gates)
    summary = _build_summary(
        pair_rows=pair_rows,
        start_rows=start_rows,
        stratum_summary=stratum_summary,
        class_summary=class_summary,
        gate_matrix=gate_matrix,
        readout_dir=readout_dir,
        output_dir=output_dir,
    )
    config = {
        "schema": "nanoclustering_g4_8_seed_start_validation_contract_config.v1",
        "readout_dir": str(readout_dir),
        "output_dir": str(output_dir),
        "start_conditions": START_CONDITIONS,
        "lane_definitions": {
            "stable_lane": "held-out split passes and all five start conditions pass",
            "conditional_lane": "held-out split passes but only some start conditions pass",
            "boundary_lane": "held-out split fails expected stratum",
        },
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(pair_rows, output_dir / CONTRACT_PAIR_ROWS_CSV)
    _write_csv(start_rows, output_dir / CONTRACT_START_ROWS_CSV)
    _write_csv(stratum_summary, output_dir / CONTRACT_STRATUM_SUMMARY_CSV)
    _write_csv(class_summary, output_dir / CONTRACT_CLASS_SUMMARY_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        stratum_summary=stratum_summary,
        class_summary=class_summary,
        gate_matrix=gate_matrix,
        pair_rows=pair_rows,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-dir", type=Path, default=DEFAULT_READOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run_contract(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
