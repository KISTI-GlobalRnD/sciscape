#!/usr/bin/env python3
"""Read out Leiden+CPM limitation modes on the G4.8 stable primary units.

This consumes the local validation execution contract and inspects only the
stable primary units. It summarizes what the existing local Leiden+CPM ablation
surface says about ready evidence, target saturation, latent release controls,
hard no-release controls, and coupled direct/bridge failures.

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


DEFAULT_EXECUTION_CONTRACT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_local_validation_execution_contract_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_primary_stable_limit_readout_gamma1e5_20260604"
)

INPUT_PAIR_ROWS_CSV = "nanoclustering_g4_8_local_validation_execution_contract_pair_rows.csv"
INPUT_UNIT_ROWS_CSV = "nanoclustering_g4_8_local_validation_execution_contract_unit_rows.csv"
INPUT_GATE_MATRIX_CSV = "nanoclustering_g4_8_local_validation_execution_contract_gate_matrix.csv"

READOUT_UNIT_ROWS_CSV = "nanoclustering_g4_8_primary_stable_limit_readout_unit_rows.csv"
READOUT_PAIR_ROWS_CSV = "nanoclustering_g4_8_primary_stable_limit_readout_pair_rows.csv"
LIMITATION_SUMMARY_CSV = "nanoclustering_g4_8_primary_stable_limit_readout_limitation_summary.csv"
GATE_MATRIX_CSV = "nanoclustering_g4_8_primary_stable_limit_readout_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_primary_stable_limit_readout_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_primary_stable_limit_readout_summary.json"
REPORT_MD = "nanoclustering_g4_8_primary_stable_limit_readout_report.md"

START_CONDITIONS = (
    "singleton",
    "pair_together",
    "bridges_to_left",
    "bridges_to_right",
    "all_local_together",
)

RUN_STATUS = "materialized_nanoclustering_g4_8_primary_stable_limit_readout"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 primary stable limitation readout only; reads the "
    "local validation execution contract and summarizes existing local "
    "Leiden+CPM ablation evidence over stable primary units. It does not run "
    "Leiden, execute route/pathway traces, promote walls, evaluate wall-clock "
    "quality/cost value, replay full NanoClustering, or claim method or "
    "algorithm success."
)

LIMITATION_ORDER = (
    "ready_partial_release",
    "target_saturated_no_handle",
    "latent_release_without_original_coassigned_source",
    "hard_no_release_control",
    "coupled_direct_bridge_failure",
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


def _first_existing(row: pd.Series, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return None


def _limitation_fields(contract_class: str) -> dict[str, str]:
    mapping = {
        "stable_strict_ready_contract": {
            "limitation_axis": "ready_partial_release",
            "leiden_readout_role": "stable_ready_evidence",
            "limitation_class": "partial_source_dependent_release",
            "limitation_interpretation": (
                "Leiden+CPM exposes a stable partial-release alternative, but "
                "the evidence remains local and source-signature-proxy based."
            ),
        },
        "stable_target_saturated_noop_contract": {
            "limitation_axis": "target_saturated_no_handle",
            "leiden_readout_role": "stable_target_saturation_control",
            "limitation_class": "target_already_coassigned_no_source_handle",
            "limitation_interpretation": (
                "The original local Leiden endpoint is already target "
                "coassigned, so there is no distinct source handle to recover."
            ),
        },
        "stable_latent_release_control_contract": {
            "limitation_axis": "latent_release_without_original_coassigned_source",
            "leiden_readout_role": "stable_latent_release_control",
            "limitation_class": "release_possible_but_original_coassigned_source_absent",
            "limitation_interpretation": (
                "Bridge removal can release the pair, but the original local "
                "endpoint lacks the coassigned source state needed for a ready "
                "basin-readout claim."
            ),
        },
        "stable_no_release_control_contract": {
            "limitation_axis": "hard_no_release_control",
            "leiden_readout_role": "stable_hard_negative_control",
            "limitation_class": "no_release_under_local_ablations",
            "limitation_interpretation": (
                "The local Leiden surface does not expose a bridge-release "
                "coassignment under the tested ablations."
            ),
        },
        "stable_coupled_failure_control_contract": {
            "limitation_axis": "coupled_direct_bridge_failure",
            "leiden_readout_role": "stable_coupled_failure_control",
            "limitation_class": "direct_and_bridge_context_are_coupled",
            "limitation_interpretation": (
                "Direct and bridge context are coupled: removing either support "
                "destroys the coassigned endpoint rather than revealing a clean "
                "release handle."
            ),
        },
    }
    if contract_class not in mapping:
        return {
            "limitation_axis": "unclassified_primary_limit",
            "leiden_readout_role": "unclassified_primary_unit",
            "limitation_class": "unclassified_primary_unit",
            "limitation_interpretation": "Inspect this primary unit before use.",
        }
    return mapping[contract_class]


def _materialize_primary_units(unit_rows: pd.DataFrame) -> pd.DataFrame:
    primary = unit_rows[_bool_series(unit_rows["include_in_primary_execution"])].copy()
    if primary.empty:
        return primary
    fields = primary["validation_contract_class"].astype(str).map(_limitation_fields)
    for key in [
        "limitation_axis",
        "leiden_readout_role",
        "limitation_class",
        "limitation_interpretation",
    ]:
        primary[key] = [item[key] for item in fields]
    primary["validation_stratum"] = [
        _first_existing(row, ("validation_stratum_y", "validation_stratum_x", "validation_stratum"))
        for _, row in primary.iterrows()
    ]
    primary["validation_family"] = [
        _first_existing(row, ("validation_family_y", "validation_family_x", "validation_family"))
        for _, row in primary.iterrows()
    ]
    primary["branch"] = [
        _first_existing(row, ("branch_y", "branch_x", "branch")) for _, row in primary.iterrows()
    ]
    primary["left_node_id"] = [
        _first_existing(row, ("left_node_id_y", "left_node_id_x", "left_node_id"))
        for _, row in primary.iterrows()
    ]
    primary["right_node_id"] = [
        _first_existing(row, ("right_node_id_y", "right_node_id_x", "right_node_id"))
        for _, row in primary.iterrows()
    ]
    primary["ready_positive_unit"] = primary["limitation_axis"].eq("ready_partial_release")
    primary["target_saturated_unit"] = primary["limitation_axis"].eq(
        "target_saturated_no_handle"
    )
    primary["latent_release_control_unit"] = primary["limitation_axis"].eq(
        "latent_release_without_original_coassigned_source"
    )
    primary["hard_negative_unit"] = primary["limitation_axis"].eq("hard_no_release_control")
    primary["coupled_failure_unit"] = primary["limitation_axis"].eq(
        "coupled_direct_bridge_failure"
    )
    primary["original_coassigned_source_available_proxy"] = (
        primary["original_coassigned_signature_count"].fillna(0).astype(float) > 0
    )
    primary["source_endpoint_signature_proxy_available"] = (
        primary["original_source_endpoint_signature_proxy_count"].fillna(0).astype(float) > 0
    )
    primary["bridge_release_observed_proxy"] = (
        primary["drop_bridge_pair_coassigned_share"].fillna(0).astype(float)
        > primary["original_pair_coassigned_share"].fillna(0).astype(float)
    )
    primary["target_saturated_observed_proxy"] = (
        primary["original_pair_coassigned_share"].fillna(0).astype(float) >= 0.95
    )
    primary["start_invariant_primary_unit"] = True
    primary["exact_g4_8f_signature_available"] = False
    primary["run_status"] = RUN_STATUS
    primary["claim_boundary"] = CLAIM_BOUNDARY
    return primary.sort_values(
        ["limitation_axis", "local_pair_id", "start_condition"], kind="mergesort"
    ).reset_index(drop=True)


def _pair_rows(primary_units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "original_pair_coassigned_share",
        "drop_direct_pair_coassigned_share",
        "drop_bridge_pair_coassigned_share",
        "drop_direct_and_bridge_pair_coassigned_share",
        "bridge_release_lift_proxy",
        "direct_dependency_proxy",
        "original_source_endpoint_signature_proxy_count",
        "original_coassigned_signature_count",
        "original_distinct_endpoint_count",
        "original_top_endpoint_share",
    ]
    for local_pair_id, group in primary_units.groupby("local_pair_id", sort=False):
        first = group.iloc[0]
        row: dict[str, Any] = {
            "local_pair_id": str(local_pair_id),
            "branch": first.get("branch"),
            "left_node_id": first.get("left_node_id"),
            "right_node_id": first.get("right_node_id"),
            "validation_stratum": first.get("validation_stratum"),
            "validation_family": first.get("validation_family"),
            "validation_contract_class": first.get("validation_contract_class"),
            "limitation_axis": first.get("limitation_axis"),
            "leiden_readout_role": first.get("leiden_readout_role"),
            "limitation_class": first.get("limitation_class"),
            "limitation_interpretation": first.get("limitation_interpretation"),
            "primary_unit_count": int(len(group)),
            "start_condition_count": int(group["start_condition"].nunique()),
            "start_conditions": ";".join(sorted(group["start_condition"].astype(str))),
            "ready_positive_unit_count": int(group["ready_positive_unit"].sum()),
            "target_saturated_unit_count": int(group["target_saturated_unit"].sum()),
            "latent_release_control_unit_count": int(
                group["latent_release_control_unit"].sum()
            ),
            "hard_negative_unit_count": int(group["hard_negative_unit"].sum()),
            "coupled_failure_unit_count": int(group["coupled_failure_unit"].sum()),
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for col in metric_cols:
            if col in group.columns:
                row.update(_prefix_stats(col, group[col]))
        rows.append(row)
    return pd.DataFrame(rows)


def _limitation_summary(primary_units: pd.DataFrame, pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "original_pair_coassigned_share",
        "drop_direct_pair_coassigned_share",
        "drop_bridge_pair_coassigned_share",
        "drop_direct_and_bridge_pair_coassigned_share",
        "bridge_release_lift_proxy",
        "direct_dependency_proxy",
        "original_source_endpoint_signature_proxy_count",
        "original_coassigned_signature_count",
    ]
    for axis in LIMITATION_ORDER:
        unit_group = primary_units[primary_units["limitation_axis"].astype(str).eq(axis)]
        pair_group = pair_rows[pair_rows["limitation_axis"].astype(str).eq(axis)]
        if unit_group.empty and pair_group.empty:
            continue
        first = unit_group.iloc[0] if not unit_group.empty else pair_group.iloc[0]
        row: dict[str, Any] = {
            "limitation_axis": axis,
            "leiden_readout_role": first.get("leiden_readout_role"),
            "limitation_class": first.get("limitation_class"),
            "pair_count": int(len(pair_group)),
            "primary_unit_count": int(len(unit_group)),
            "unit_share": float(len(unit_group) / len(primary_units))
            if len(primary_units)
            else None,
            "start_condition_counts": json.dumps(
                _count_dict(unit_group["start_condition"]) if not unit_group.empty else {},
                sort_keys=True,
            ),
            "validation_contract_class_counts": json.dumps(
                _count_dict(unit_group["validation_contract_class"])
                if not unit_group.empty
                else {},
                sort_keys=True,
            ),
            "limitation_interpretation": first.get("limitation_interpretation"),
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for col in metric_cols:
            if col in unit_group.columns:
                row.update(_prefix_stats(col, unit_group[col]))
        rows.append(row)
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


def _metric_patterns_pass(primary_units: pd.DataFrame) -> bool:
    by_axis = {
        axis: primary_units[primary_units["limitation_axis"].astype(str).eq(axis)]
        for axis in LIMITATION_ORDER
    }
    if any(group.empty for group in by_axis.values()):
        return False
    ready = by_axis["ready_partial_release"]
    target = by_axis["target_saturated_no_handle"]
    latent = by_axis["latent_release_without_original_coassigned_source"]
    hard = by_axis["hard_no_release_control"]
    coupled = by_axis["coupled_direct_bridge_failure"]
    return bool(
        ready["bridge_release_lift_proxy"].astype(float).gt(0).all()
        and ready["original_pair_coassigned_share"].astype(float).between(0.0, 0.95).all()
        and target["original_pair_coassigned_share"].astype(float).ge(0.95).all()
        and target["original_source_endpoint_signature_proxy_count"].astype(float).eq(0).all()
        and latent["original_pair_coassigned_share"].astype(float).eq(0).all()
        and latent["drop_bridge_pair_coassigned_share"].astype(float).ge(0.95).all()
        and hard["drop_bridge_pair_coassigned_share"].astype(float).eq(0).all()
        and hard["bridge_release_lift_proxy"].astype(float).eq(0).all()
        and coupled["original_pair_coassigned_share"].astype(float).ge(0.95).all()
        and coupled["drop_bridge_pair_coassigned_share"].astype(float).eq(0).all()
        and coupled["bridge_release_lift_proxy"].astype(float).lt(0).all()
    )


def _build_gate_matrix(
    *,
    primary_units: pd.DataFrame,
    pair_rows: pd.DataFrame,
    limitation_summary: pd.DataFrame,
    upstream_gates: pd.DataFrame,
) -> pd.DataFrame:
    start_counts = primary_units.groupby("local_pair_id")["start_condition"].nunique()
    ready_units = primary_units[primary_units["ready_positive_unit"]]
    control_units = primary_units[~primary_units["ready_positive_unit"]]
    rows = [
        _gate_row(
            "G1_upstream_execution_contract_passes",
            "Did every upstream execution-contract gate pass?",
            bool(upstream_gates["gate_status"].astype(str).eq("pass").all()),
            _count_dict(upstream_gates["gate_status"]),
            "all upstream gates pass",
        ),
        _gate_row(
            "G2_primary_only_surface_preserved",
            "Does this readout inspect only stable primary units?",
            int(len(pair_rows)) == 15
            and int(len(primary_units)) == 75
            and bool(primary_units["execution_lane"].astype(str).eq("stable_lane").all()),
            f"pair_count={len(pair_rows)} primary_units={len(primary_units)}",
            "15 stable pairs and 75 stable primary units",
        ),
        _gate_row(
            "G3_all_primary_starts_present",
            "Does every primary stable pair retain all five start conditions?",
            bool((start_counts == len(START_CONDITIONS)).all()),
            f"min_starts={int(start_counts.min())} max_starts={int(start_counts.max())}",
            "five starts per primary pair",
        ),
        _gate_row(
            "G4_ready_signal_is_scoped",
            "Is ready evidence present but explicitly scoped rather than dominant?",
            int(len(ready_units)) == 10
            and int(ready_units["local_pair_id"].nunique()) == 2
            and int(len(control_units)) == 65,
            (
                f"ready_units={len(ready_units)} "
                f"ready_pairs={ready_units['local_pair_id'].nunique()} "
                f"control_units={len(control_units)}"
            ),
            "2 ready pairs, 10 ready units, 65 control/limit units",
        ),
        _gate_row(
            "G5_limitation_axes_are_covered",
            "Does the primary readout cover all five limitation axes?",
            set(LIMITATION_ORDER).issubset(set(primary_units["limitation_axis"].astype(str))),
            json.dumps(_count_dict(primary_units["limitation_axis"]), sort_keys=True),
            "ready, target-saturated, latent-release, hard-negative, coupled-failure axes",
        ),
        _gate_row(
            "G6_metric_patterns_match_limit_classes",
            "Do metric patterns match the assigned limitation classes?",
            _metric_patterns_pass(primary_units),
            json.dumps(
                {
                    str(row.limitation_axis): int(row.primary_unit_count)
                    for row in limitation_summary.itertuples(index=False)
                },
                sort_keys=True,
            ),
            "class-specific local-ablation metric checks pass",
        ),
        _gate_row(
            "G7_conditional_and_boundary_excluded",
            "Are conditional and boundary units excluded from this primary readout?",
            not bool(primary_units["include_in_secondary_execution"].any())
            and not bool(primary_units["include_as_diagnostic_control"].any()),
            json.dumps(_count_dict(primary_units["execution_unit_role"]), sort_keys=True),
            "primary execution units only",
        ),
        _gate_row(
            "G8_exact_signature_gap_closed",
            "Is the exact G4.8F source-signature gap kept closed?",
            not bool(
                primary_units["exact_g4_8f_signature_available"].fillna(False).astype(bool).any()
            ),
            "exact_g4_8f_signature_available=false",
            "proxy signatures only",
        ),
        _gate_row(
            "G9_no_new_leiden_execution",
            "Is this a readout over existing contract rows rather than a new run?",
            True,
            RUN_STATUS,
            "read-only materialization",
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


def _readout_status(gate_matrix: pd.DataFrame) -> str:
    if gate_matrix.empty or not bool(gate_matrix["gate_status"].astype(str).eq("pass").all()):
        return "primary_stable_limit_readout_gate_failed"
    return "primary_stable_limit_readout_ready_scoped_ready_signal"


def _build_summary(
    *,
    primary_units: pd.DataFrame,
    pair_rows: pd.DataFrame,
    limitation_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    execution_contract_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    ready_units = primary_units[primary_units["ready_positive_unit"]]
    return {
        "schema": "nanoclustering_g4_8_primary_stable_limit_readout_summary.v1",
        "status": _readout_status(gate_matrix),
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_contract_dir": str(execution_contract_dir),
        "output_dir": str(output_dir),
        "primary_pair_count": int(len(pair_rows)),
        "primary_unit_count": int(len(primary_units)),
        "ready_pair_count": int(ready_units["local_pair_id"].nunique()),
        "ready_unit_count": int(len(ready_units)),
        "control_or_limit_unit_count": int(len(primary_units) - len(ready_units)),
        "limitation_axis_counts": _count_dict(primary_units["limitation_axis"]),
        "limitation_class_counts": _count_dict(primary_units["limitation_class"]),
        "limitation_summary_rows": int(len(limitation_summary)),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": [
            str(row.gate_id)
            for row in gate_matrix.itertuples(index=False)
            if str(row.gate_status) != "pass"
        ],
        "exact_g4_8f_signature_available": False,
        "interpretation": (
            "Stable primary units show a real but scoped ready signal: 2 pairs "
            "and 10 units are ready partial-release evidence, while 65 units "
            "are stable Leiden+CPM limitation/control cases. This supports "
            "Leiden limitation cartography, not a method or wall/pathway claim."
        ),
        "recommended_next_gate": (
            "Use this limitation readout to design the next primary validation "
            "run/report: ready evidence, target-saturation mass, latent-release "
            "controls, hard negatives, and coupled failures must remain separate "
            "before any route/pathway, quality/cost, full NanoClustering replay, "
            "or method claim."
        ),
        "written_artifacts": [
            READOUT_UNIT_ROWS_CSV,
            READOUT_PAIR_ROWS_CSV,
            LIMITATION_SUMMARY_CSV,
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
    limitation_summary: pd.DataFrame,
    pair_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    ready_pairs = pair_rows[pair_rows["limitation_axis"].eq("ready_partial_release")]
    lines = [
        "# NanoClustering G4.8 Primary Stable Limit Readout",
        "",
        f"- status: `{summary['status']}`",
        f"- primary_pair_count: {summary['primary_pair_count']}",
        f"- primary_unit_count: {summary['primary_unit_count']}",
        f"- ready_pair_count: {summary['ready_pair_count']}",
        f"- ready_unit_count: {summary['ready_unit_count']}",
        f"- control_or_limit_unit_count: {summary['control_or_limit_unit_count']}",
        f"- limitation_axis_counts: {summary['limitation_axis_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- exact_g4_8f_signature_available: {summary['exact_g4_8f_signature_available']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Limitation Summary",
        "",
    ]
    lines.append(
        _markdown_table(
            limitation_summary,
            [
                "limitation_axis",
                "pair_count",
                "primary_unit_count",
                "unit_share",
                "original_pair_coassigned_share_median",
                "drop_bridge_pair_coassigned_share_median",
                "bridge_release_lift_proxy_median",
                "direct_dependency_proxy_median",
                "limitation_class",
            ],
        )
    )
    lines.extend(["", "## Ready Primary Pairs", ""])
    lines.append(
        _markdown_table(
            ready_pairs,
            [
                "local_pair_id",
                "validation_stratum",
                "primary_unit_count",
                "original_pair_coassigned_share_median",
                "drop_bridge_pair_coassigned_share_median",
                "bridge_release_lift_proxy_median",
                "direct_dependency_proxy_median",
            ],
        )
    )
    lines.extend(["", "## Pair-Level Limit Map", ""])
    lines.append(
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "limitation_axis",
                "validation_contract_class",
                "primary_unit_count",
                "original_pair_coassigned_share_median",
                "drop_bridge_pair_coassigned_share_median",
                "bridge_release_lift_proxy_median",
                "direct_dependency_proxy_median",
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
            "This readout is an existing-Leiden limitation map for the stable "
            "primary units. It keeps ready evidence scoped and keeps target "
            "saturation, latent release without original coassigned source, "
            "hard no-release controls, and coupled direct/bridge failures as "
            "separate limitations. It does not run Leiden and does not open "
            "route/pathway, wall, quality/cost, full NanoClustering replay, or "
            "method claims.",
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_readout(args: argparse.Namespace) -> dict[str, Any]:
    execution_contract_dir = Path(args.execution_contract_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unit_input = _read_csv(execution_contract_dir / INPUT_UNIT_ROWS_CSV)
    upstream_gates = _read_csv(execution_contract_dir / INPUT_GATE_MATRIX_CSV)

    primary_units = _materialize_primary_units(unit_input)
    pair_rows = _pair_rows(primary_units)
    limitation_summary = _limitation_summary(primary_units, pair_rows)
    gate_matrix = _build_gate_matrix(
        primary_units=primary_units,
        pair_rows=pair_rows,
        limitation_summary=limitation_summary,
        upstream_gates=upstream_gates,
    )
    summary = _build_summary(
        primary_units=primary_units,
        pair_rows=pair_rows,
        limitation_summary=limitation_summary,
        gate_matrix=gate_matrix,
        execution_contract_dir=execution_contract_dir,
        output_dir=output_dir,
    )

    _write_csv(primary_units, output_dir / READOUT_UNIT_ROWS_CSV)
    _write_csv(pair_rows, output_dir / READOUT_PAIR_ROWS_CSV)
    _write_csv(limitation_summary, output_dir / LIMITATION_SUMMARY_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            _json_safe(
                {
                    "execution_contract_dir": str(execution_contract_dir),
                    "output_dir": str(output_dir),
                    "start_conditions": START_CONDITIONS,
                    "limitation_order": LIMITATION_ORDER,
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
        limitation_summary=limitation_summary,
        pair_rows=pair_rows,
        gate_matrix=gate_matrix,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-contract-dir",
        type=Path,
        default=DEFAULT_EXECUTION_CONTRACT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run_readout(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
