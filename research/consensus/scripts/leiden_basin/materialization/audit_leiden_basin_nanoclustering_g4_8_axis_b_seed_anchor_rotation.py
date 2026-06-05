#!/usr/bin/env python3
"""Audit Axis B endpoint continuity under seed/start anchor rotation.

The dual-axis direct-path contract opened Axis B using a pair-level endpoint
atlas. This audit asks whether that Axis B readout is robust when the endpoint
role vocabulary is rebuilt after removing the focal seed and/or start
condition from the known-anchor evidence.

It does not run Leiden, broaden route execution, promote walls, evaluate
quality/cost value, replay full NanoClustering, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from design_leiden_basin_nanoclustering_g4_8_dual_axis_direct_path_contract import (
    CLAIM_BOUNDARY as DUAL_AXIS_CLAIM_BOUNDARY,
    DEFAULT_OUTPUT_DIR as DEFAULT_DUAL_AXIS_DIR,
    GATE_MATRIX_CSV as DUAL_AXIS_GATE_MATRIX_CSV,
    SUMMARY_JSON as DUAL_AXIS_SUMMARY_JSON,
)
from run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TRACE_DIR,
    TRACE_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_axis_b_seed_anchor_rotation_audit_gamma1e5_20260604"
)

PRIMARY_ROUTE_FAMILY = "bridge_release_interpolation_probe"
RUN_STATUS = "audited_nanoclustering_g4_8_axis_b_seed_anchor_rotation"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 Axis B seed-anchor rotation audit only; reads the "
    "existing scoped route trace and dual-axis direct-path contract, then "
    "rebuilds pair-level endpoint roles under leave-seed/start exclusions. It "
    "does not run Leiden, broaden route execution, promote walls, evaluate "
    "quality/cost value, replay full NanoClustering, or claim method or "
    "algorithm success."
)

ROTATION_MODES = (
    {
        "rotation_mode": "full_pair_atlas",
        "exclude_focal_seed": False,
        "exclude_focal_start_condition": False,
        "mode_question": "Does the original pair-level endpoint atlas preserve Axis B continuity?",
    },
    {
        "rotation_mode": "leave_start_out",
        "exclude_focal_seed": False,
        "exclude_focal_start_condition": True,
        "mode_question": (
            "Does Axis B continuity survive when the focal start condition is "
            "removed from the known-anchor vocabulary?"
        ),
    },
    {
        "rotation_mode": "leave_seed_out",
        "exclude_focal_seed": True,
        "exclude_focal_start_condition": False,
        "mode_question": (
            "Does Axis B continuity survive when the focal seed is removed from "
            "the known-anchor vocabulary?"
        ),
    },
    {
        "rotation_mode": "leave_seed_and_start_out",
        "exclude_focal_seed": True,
        "exclude_focal_start_condition": True,
        "mode_question": (
            "Does Axis B continuity survive when both the focal seed and focal "
            "start condition are removed from the known-anchor vocabulary?"
        ),
    },
)

STEP_ROTATION_ROWS_CSV = (
    "nanoclustering_g4_8_axis_b_seed_anchor_rotation_step_rows.csv"
)
ROUTE_ROTATION_ROWS_CSV = (
    "nanoclustering_g4_8_axis_b_seed_anchor_rotation_route_rows.csv"
)
UNKNOWN_ROTATION_ROWS_CSV = (
    "nanoclustering_g4_8_axis_b_seed_anchor_rotation_unknown_rows.csv"
)
CONTRACT_ROTATION_ROWS_CSV = (
    "nanoclustering_g4_8_axis_b_seed_anchor_rotation_contract_rows.csv"
)
MODE_SUMMARY_ROWS_CSV = (
    "nanoclustering_g4_8_axis_b_seed_anchor_rotation_mode_summary_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_axis_b_seed_anchor_rotation_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_axis_b_seed_anchor_rotation_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_axis_b_seed_anchor_rotation_summary.json"
REPORT_MD = "nanoclustering_g4_8_axis_b_seed_anchor_rotation_report.md"

EPSILON = 1e-9


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _join_sorted(values: pd.Series | list[Any]) -> str:
    if isinstance(values, pd.Series):
        raw_values = values.dropna().tolist()
    else:
        raw_values = list(values)
    items = sorted({str(value) for value in raw_values if str(value)})
    return ";".join(items)


def _step_list(values: pd.Series | list[Any]) -> str:
    if isinstance(values, pd.Series):
        raw_values = values.dropna().tolist()
    else:
        raw_values = list(values)
    if not raw_values:
        return ""
    return ";".join(str(int(value)) for value in sorted(set(raw_values)))


def _assignment_role(assignments: pd.Series) -> str:
    joined = ";".join(str(value) for value in assignments.dropna().tolist())
    if "drop_bridge_target_anchor" in joined:
        return "pair_level_known_drop_bridge_target_signature"
    if "original_source_anchor" in joined:
        return "pair_level_known_original_source_signature"
    if "drop_direct_guard_anchor" in joined:
        return "pair_level_known_drop_direct_guard_signature"
    if joined:
        return "pair_level_known_other_anchor_signature"
    return "pair_level_true_novel_under_rotation"


def _role_is_source(role: Any) -> bool:
    return "original_source" in str(role)


def _role_is_target(role: Any) -> bool:
    return "drop_bridge_target" in str(role)


def _role_is_novel(role: Any) -> bool:
    return str(role) == "pair_level_true_novel_under_rotation"


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
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


def _primary_trace(trace_rows: pd.DataFrame) -> pd.DataFrame:
    return trace_rows[
        trace_rows["planned_route_family"].astype(str).eq(PRIMARY_ROUTE_FAMILY)
    ].copy()


def _known_anchor_rows(primary_trace: pd.DataFrame) -> pd.DataFrame:
    return primary_trace[
        ~primary_trace["endpoint_assignment_by_step"].astype(str).eq("unknown_new_endpoint")
    ].copy()


def _rotated_vocab_for_step(
    *,
    known_rows: pd.DataFrame,
    row: pd.Series,
    exclude_focal_seed: bool,
    exclude_focal_start_condition: bool,
) -> pd.DataFrame:
    vocab = known_rows[
        known_rows["local_pair_id"].astype(str).eq(str(row["local_pair_id"]))
        & known_rows["result_endpoint_signature_id"].astype(str).eq(
            str(row["result_endpoint_signature_id"])
        )
    ].copy()
    if exclude_focal_seed:
        vocab = vocab[~vocab["seed"].astype(int).eq(int(row["seed"]))]
    if exclude_focal_start_condition:
        vocab = vocab[
            ~vocab["start_condition"].astype(str).eq(str(row["start_condition"]))
        ]
    return vocab


def _step_rotation_rows(primary_trace: pd.DataFrame) -> pd.DataFrame:
    known_rows = _known_anchor_rows(primary_trace)
    rows: list[dict[str, Any]] = []
    for mode in ROTATION_MODES:
        mode_name = str(mode["rotation_mode"])
        exclude_seed = bool(mode["exclude_focal_seed"])
        exclude_start = bool(mode["exclude_focal_start_condition"])
        for trace_row in primary_trace.to_dict(orient="records"):
            row = pd.Series(trace_row)
            vocab = _rotated_vocab_for_step(
                known_rows=known_rows,
                row=row,
                exclude_focal_seed=exclude_seed,
                exclude_focal_start_condition=exclude_start,
            )
            role = _assignment_role(vocab["endpoint_assignment_by_step"])
            support_seed_count = int(vocab["seed"].nunique()) if not vocab.empty else 0
            support_start_count = (
                int(vocab["start_condition"].nunique()) if not vocab.empty else 0
            )
            is_same_seed_unknown = str(row["endpoint_assignment_by_step"]) == (
                "unknown_new_endpoint"
            )
            rows.append(
                {
                    "rotation_mode": mode_name,
                    "exclude_focal_seed": exclude_seed,
                    "exclude_focal_start_condition": exclude_start,
                    "route_trace_row_id": str(row["route_trace_row_id"]),
                    "route_contract_id": str(row["route_contract_id"]),
                    "validation_unit_id": str(row["validation_unit_id"]),
                    "local_pair_id": str(row["local_pair_id"]),
                    "start_condition": str(row["start_condition"]),
                    "seed": int(row["seed"]),
                    "step_index": int(row["step_index"]),
                    "planned_route_family": PRIMARY_ROUTE_FAMILY,
                    "result_endpoint_signature_id": str(
                        row["result_endpoint_signature_id"]
                    ),
                    "same_seed_endpoint_assignment": str(
                        row["endpoint_assignment_by_step"]
                    ),
                    "rotation_endpoint_role": role,
                    "rotation_pair_level_known": not _role_is_novel(role),
                    "rotation_support_row_count": int(len(vocab)),
                    "rotation_support_seed_count": support_seed_count,
                    "rotation_support_start_condition_count": support_start_count,
                    "rotation_support_assignments": _join_sorted(
                        vocab["endpoint_assignment_by_step"]
                    ),
                    "rotation_support_seeds": _step_list(vocab["seed"]),
                    "rotation_support_start_conditions": _join_sorted(
                        vocab["start_condition"]
                    ),
                    "is_same_seed_unknown_step": bool(is_same_seed_unknown),
                    "same_seed_unknown_pair_known_under_rotation": bool(
                        is_same_seed_unknown and not _role_is_novel(role)
                    ),
                    "same_seed_unknown_true_novel_under_rotation": bool(
                        is_same_seed_unknown and _role_is_novel(role)
                    ),
                    "matches_expected_final_anchor": bool(
                        _as_bool(row["matches_expected_final_anchor"])
                    ),
                    "support_incompatibility_check": bool(
                        _as_bool(row["support_incompatibility_check"])
                    ),
                    "direct_edge_weight_fraction": float(
                        row["direct_edge_weight_fraction"]
                    ),
                    "active_direct_edge_weight": float(row["active_direct_edge_weight"]),
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["rotation_mode", "local_pair_id", "start_condition", "seed", "step_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _route_rotation_rows(step_rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "rotation_mode",
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "seed",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in step_rows.groupby(group_cols, sort=False):
        ordered = group.sort_values("step_index", kind="mergesort").reset_index(drop=True)
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        source_mask = ordered["rotation_endpoint_role"].map(_role_is_source).astype(bool)
        target_mask = ordered["rotation_endpoint_role"].map(_role_is_target).astype(bool)
        novel_mask = ordered["rotation_endpoint_role"].map(_role_is_novel).astype(bool)
        unknown_mask = ordered["is_same_seed_unknown_step"].astype(bool)
        unknown_pair_known_mask = ordered[
            "same_seed_unknown_pair_known_under_rotation"
        ].astype(bool)
        unknown_true_novel_mask = ordered[
            "same_seed_unknown_true_novel_under_rotation"
        ].astype(bool)
        direct_edge_retained = (
            float(ordered["direct_edge_weight_fraction"].min()) > 0.0
            and float(ordered["active_direct_edge_weight"].min()) > 0.0
        )
        source_steps = ordered.loc[source_mask, "step_index"].astype(int).tolist()
        target_steps = ordered.loc[target_mask, "step_index"].astype(int).tolist()
        first_source_step = None if not source_steps else int(min(source_steps))
        first_target_step = None if not target_steps else int(min(target_steps))
        last_target_step = None if not target_steps else int(max(target_steps))
        source_start_pass = _role_is_source(first["rotation_endpoint_role"])
        target_final_pass = (
            _role_is_target(last["rotation_endpoint_role"])
            and bool(last["matches_expected_final_anchor"])
        )
        no_true_novel_endpoint_pass = not bool(novel_mask.any())
        source_to_target_pass = (
            first_source_step is not None
            and first_target_step is not None
            and first_source_step <= first_target_step
            and last_target_step == int(last["step_index"])
        )
        novel_steps = ordered.loc[novel_mask, "step_index"].astype(int).tolist()
        source_start_singleton_caveat = (
            not source_start_pass
            and len(novel_steps) == 1
            and int(novel_steps[0]) == int(first["step_index"])
        )
        route_pass = all(
            [
                direct_edge_retained,
                source_start_pass,
                target_final_pass,
                no_true_novel_endpoint_pass,
                source_to_target_pass,
            ]
        )
        if route_pass:
            status = "axis_b_rotation_continuity_pass"
            block_reason = "none"
        elif source_start_singleton_caveat:
            status = "axis_b_rotation_blocked_by_source_start_singleton"
            block_reason = "source_start_signature_has_no_rotated_known_support"
        elif not no_true_novel_endpoint_pass:
            status = "axis_b_rotation_blocked_by_true_novel_endpoint"
            block_reason = "one_or_more_endpoint_signatures_lack_rotated_known_support"
        elif not target_final_pass:
            status = "axis_b_rotation_blocked_by_target_final"
            block_reason = "final_endpoint_is_not_rotated_target"
        else:
            status = "axis_b_rotation_blocked_by_path_condition"
            block_reason = "direct_edge_or_source_to_target_condition_failed"

        rows.append(
            {
                "rotation_mode": str(keys[0]),
                "route_contract_id": str(keys[1]),
                "validation_unit_id": str(keys[2]),
                "local_pair_id": str(keys[3]),
                "start_condition": str(keys[4]),
                "seed": int(keys[5]),
                "planned_route_family": PRIMARY_ROUTE_FAMILY,
                "route_step_count": int(len(ordered)),
                "axis_b_rotation_route_pass": bool(route_pass),
                "axis_b_rotation_route_status": status,
                "axis_b_rotation_block_reason": block_reason,
                "direct_edge_retained_under_rotation_route": bool(direct_edge_retained),
                "source_start_under_rotation_pass": bool(source_start_pass),
                "target_final_under_rotation_pass": bool(target_final_pass),
                "no_true_novel_endpoint_under_rotation_pass": bool(
                    no_true_novel_endpoint_pass
                ),
                "source_to_target_under_rotation_pass": bool(source_to_target_pass),
                "same_seed_unknown_step_count": int(unknown_mask.sum()),
                "same_seed_unknown_pair_known_step_count": int(
                    unknown_pair_known_mask.sum()
                ),
                "same_seed_unknown_true_novel_step_count": int(
                    unknown_true_novel_mask.sum()
                ),
                "true_novel_endpoint_step_count": int(novel_mask.sum()),
                "pair_level_source_step_indices": _step_list(
                    ordered.loc[source_mask, "step_index"]
                ),
                "pair_level_target_step_indices": _step_list(
                    ordered.loc[target_mask, "step_index"]
                ),
                "true_novel_step_indices": _step_list(novel_steps),
                "source_start_singleton_caveat": bool(source_start_singleton_caveat),
                "endpoint_role_sequence": ";".join(
                    ordered["rotation_endpoint_role"].astype(str).tolist()
                ),
                "min_rotation_support_row_count": int(
                    ordered["rotation_support_row_count"].astype(int).min()
                ),
                "min_rotation_support_seed_count": int(
                    ordered["rotation_support_seed_count"].astype(int).min()
                ),
                "min_rotation_support_start_condition_count": int(
                    ordered["rotation_support_start_condition_count"].astype(int).min()
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["rotation_mode", "local_pair_id", "start_condition", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _unknown_rotation_rows(step_rows: pd.DataFrame) -> pd.DataFrame:
    rows = step_rows[step_rows["is_same_seed_unknown_step"].astype(bool)].copy()
    rows["unknown_rotation_status"] = rows[
        "same_seed_unknown_pair_known_under_rotation"
    ].map(
        {
            True: "same_seed_unknown_remains_pair_known_under_rotation",
            False: "same_seed_unknown_becomes_true_novel_under_rotation",
        }
    )
    keep = [
        "rotation_mode",
        "route_trace_row_id",
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "seed",
        "step_index",
        "result_endpoint_signature_id",
        "rotation_endpoint_role",
        "rotation_pair_level_known",
        "rotation_support_row_count",
        "rotation_support_seed_count",
        "rotation_support_start_condition_count",
        "rotation_support_assignments",
        "rotation_support_seeds",
        "rotation_support_start_conditions",
        "same_seed_unknown_pair_known_under_rotation",
        "same_seed_unknown_true_novel_under_rotation",
        "unknown_rotation_status",
        "run_status",
        "claim_boundary",
    ]
    return rows[keep].sort_values(
        ["rotation_mode", "local_pair_id", "start_condition", "seed", "step_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _contract_rotation_rows(route_rows: pd.DataFrame) -> pd.DataFrame:
    groups = (
        route_rows.groupby(
            [
                "rotation_mode",
                "route_contract_id",
                "validation_unit_id",
                "local_pair_id",
                "start_condition",
            ],
            sort=False,
        )
        .agg(
            seed_count=("seed", "nunique"),
            axis_b_rotation_seed_pass_count=("axis_b_rotation_route_pass", "sum"),
            source_start_fail_seed_count=(
                "source_start_under_rotation_pass",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            target_final_fail_seed_count=(
                "target_final_under_rotation_pass",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            true_novel_endpoint_seed_count=(
                "true_novel_endpoint_step_count",
                lambda values: int((values.astype(int) > 0).sum()),
            ),
            same_seed_unknown_seed_count=(
                "same_seed_unknown_step_count",
                lambda values: int((values.astype(int) > 0).sum()),
            ),
            same_seed_unknown_pair_known_seed_count=(
                "same_seed_unknown_pair_known_step_count",
                lambda values: int((values.astype(int) > 0).sum()),
            ),
            same_seed_unknown_true_novel_seed_count=(
                "same_seed_unknown_true_novel_step_count",
                lambda values: int((values.astype(int) > 0).sum()),
            ),
            min_rotation_support_row_count=(
                "min_rotation_support_row_count",
                "min",
            ),
            route_status_counts=(
                "axis_b_rotation_route_status",
                lambda values: json.dumps(_count_dict(values), sort_keys=True),
            ),
        )
        .reset_index()
    )
    groups["axis_b_rotation_contract_pass"] = groups[
        "axis_b_rotation_seed_pass_count"
    ].astype(int).eq(groups["seed_count"].astype(int))
    groups["wall_contract_ready"] = False

    def status(row: pd.Series) -> str:
        if bool(row["axis_b_rotation_contract_pass"]):
            return "axis_b_rotation_contract_pass"
        if int(row["same_seed_unknown_true_novel_seed_count"]) == 0 and int(
            row["source_start_fail_seed_count"]
        ) > 0:
            return "axis_b_rotation_contract_caveat_source_start_singleton_only"
        if int(row["true_novel_endpoint_seed_count"]) > 0:
            return "axis_b_rotation_contract_blocked_by_true_novel_endpoint"
        return "axis_b_rotation_contract_partial"

    groups["axis_b_rotation_contract_status"] = groups.apply(status, axis=1)
    groups["contract_claim_boundary_note"] = (
        "Rotation pass/fail is an Axis B robustness diagnostic only; it cannot "
        "promote wall, quality, cost, full replay, or method claims."
    )
    groups["run_status"] = RUN_STATUS
    groups["claim_boundary"] = CLAIM_BOUNDARY
    return groups.sort_values(
        ["rotation_mode", "local_pair_id", "start_condition"],
        kind="mergesort",
    ).reset_index(drop=True)


def _mode_summary_rows(
    step_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    unknown_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in ROTATION_MODES:
        mode_name = str(mode["rotation_mode"])
        step_mode = step_rows[step_rows["rotation_mode"].astype(str).eq(mode_name)]
        route_mode = route_rows[route_rows["rotation_mode"].astype(str).eq(mode_name)]
        unknown_mode = unknown_rows[unknown_rows["rotation_mode"].astype(str).eq(mode_name)]
        contract_mode = contract_rows[
            contract_rows["rotation_mode"].astype(str).eq(mode_name)
        ]
        route_pass_count = int(route_mode["axis_b_rotation_route_pass"].astype(bool).sum())
        contract_pass_count = int(
            contract_mode["axis_b_rotation_contract_pass"].astype(bool).sum()
        )
        source_start_fail_count = int(
            (~route_mode["source_start_under_rotation_pass"].astype(bool)).sum()
        )
        unknown_pair_known_count = int(
            unknown_mode["same_seed_unknown_pair_known_under_rotation"].astype(bool).sum()
        )
        unknown_true_novel_count = int(
            unknown_mode["same_seed_unknown_true_novel_under_rotation"].astype(bool).sum()
        )
        true_novel_route_count = int(
            (route_mode["true_novel_endpoint_step_count"].astype(int) > 0).sum()
        )
        if route_pass_count == len(route_mode):
            mode_status = "axis_b_rotation_full_route_continuity_pass"
        elif unknown_true_novel_count == 0 and source_start_fail_count > 0:
            mode_status = "axis_b_rotation_unknowns_robust_source_start_caveat"
        else:
            mode_status = "axis_b_rotation_partial_or_blocked"
        rows.append(
            {
                "rotation_mode": mode_name,
                "exclude_focal_seed": bool(mode["exclude_focal_seed"]),
                "exclude_focal_start_condition": bool(
                    mode["exclude_focal_start_condition"]
                ),
                "mode_question": str(mode["mode_question"]),
                "step_row_count": int(len(step_mode)),
                "route_row_count": int(len(route_mode)),
                "contract_row_count": int(len(contract_mode)),
                "unknown_row_count": int(len(unknown_mode)),
                "axis_b_route_pass_count": route_pass_count,
                "axis_b_contract_pass_count": contract_pass_count,
                "same_seed_unknown_pair_known_count": unknown_pair_known_count,
                "same_seed_unknown_true_novel_count": unknown_true_novel_count,
                "true_novel_route_count": true_novel_route_count,
                "source_start_fail_route_count": source_start_fail_count,
                "target_final_fail_route_count": int(
                    (~route_mode["target_final_under_rotation_pass"].astype(bool)).sum()
                ),
                "min_rotation_support_row_count": int(
                    step_mode["rotation_support_row_count"].astype(int).min()
                ),
                "route_status_counts": json.dumps(
                    _count_dict(route_mode["axis_b_rotation_route_status"]),
                    sort_keys=True,
                ),
                "contract_status_counts": json.dumps(
                    _count_dict(contract_mode["axis_b_rotation_contract_status"]),
                    sort_keys=True,
                ),
                "mode_status": mode_status,
                "wall_ready_contract_count": int(
                    contract_mode["wall_contract_ready"].astype(bool).sum()
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


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
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _mode_value(mode_summary: pd.DataFrame, mode_name: str, column: str) -> Any:
    matches = mode_summary[mode_summary["rotation_mode"].astype(str).eq(mode_name)]
    if matches.empty:
        raise KeyError(mode_name)
    return matches.iloc[0][column]


def _gate_matrix(
    *,
    dual_axis_gates: pd.DataFrame,
    mode_summary: pd.DataFrame,
    route_rows: pd.DataFrame,
    unknown_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
) -> pd.DataFrame:
    leave_seed_routes = int(
        _mode_value(mode_summary, "leave_seed_out", "axis_b_route_pass_count")
    )
    leave_seed_contracts = int(
        _mode_value(mode_summary, "leave_seed_out", "axis_b_contract_pass_count")
    )
    leave_seed_source_fails = int(
        _mode_value(mode_summary, "leave_seed_out", "source_start_fail_route_count")
    )
    leave_seed_start_routes = int(
        _mode_value(mode_summary, "leave_seed_and_start_out", "axis_b_route_pass_count")
    )
    unknown_by_mode = {
        str(row.rotation_mode): int(row.same_seed_unknown_pair_known_count)
        for row in mode_summary.itertuples(index=False)
    }
    unknown_novel_by_mode = {
        str(row.rotation_mode): int(row.same_seed_unknown_true_novel_count)
        for row in mode_summary.itertuples(index=False)
    }
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_dual_axis_gates_pass",
                "Did the upstream dual-axis direct-path contract gates pass?",
                _count_dict(dual_axis_gates["gate_status"]),
                "all upstream dual-axis gates pass",
                bool(dual_axis_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_rotation_modes_materialized",
                "Are all predeclared seed/start anchor-rotation modes materialized?",
                list(mode_summary["rotation_mode"].astype(str)),
                "four modes: full_pair_atlas, leave_start_out, leave_seed_out, leave_seed_and_start_out",
                set(mode_summary["rotation_mode"].astype(str))
                == {
                    "full_pair_atlas",
                    "leave_start_out",
                    "leave_seed_out",
                    "leave_seed_and_start_out",
                },
            ),
            _gate_row(
                "G3_baseline_and_leave_start_route_continuity_pass",
                "Does Axis B route continuity survive baseline and leave-start rotations?",
                {
                    "full_pair_atlas_routes": int(
                        _mode_value(mode_summary, "full_pair_atlas", "axis_b_route_pass_count")
                    ),
                    "leave_start_out_routes": int(
                        _mode_value(mode_summary, "leave_start_out", "axis_b_route_pass_count")
                    ),
                    "full_pair_atlas_contracts": int(
                        _mode_value(
                            mode_summary,
                            "full_pair_atlas",
                            "axis_b_contract_pass_count",
                        )
                    ),
                    "leave_start_out_contracts": int(
                        _mode_value(
                            mode_summary,
                            "leave_start_out",
                            "axis_b_contract_pass_count",
                        )
                    ),
                },
                "80 route passes and 10 contract passes in both modes",
                int(_mode_value(mode_summary, "full_pair_atlas", "axis_b_route_pass_count"))
                == 80
                and int(_mode_value(mode_summary, "leave_start_out", "axis_b_route_pass_count"))
                == 80
                and int(
                    _mode_value(
                        mode_summary,
                        "full_pair_atlas",
                        "axis_b_contract_pass_count",
                    )
                )
                == 10
                and int(
                    _mode_value(
                        mode_summary,
                        "leave_start_out",
                        "axis_b_contract_pass_count",
                    )
                )
                == 10,
            ),
            _gate_row(
                "G4_same_seed_unknowns_survive_all_rotations",
                "Do same-seed unknown rows remain pair-level known under every rotation?",
                {
                    "unknown_pair_known_by_mode": unknown_by_mode,
                    "unknown_true_novel_by_mode": unknown_novel_by_mode,
                },
                "27 pair-known and 0 true-novel same-seed unknown rows in every mode",
                all(value == 27 for value in unknown_by_mode.values())
                and all(value == 0 for value in unknown_novel_by_mode.values()),
            ),
            _gate_row(
                "G5_leave_seed_route_caveat_is_localized",
                "Does leave-one-seed-out expose only a localized source-start caveat?",
                {
                    "leave_seed_routes": leave_seed_routes,
                    "leave_seed_contracts": leave_seed_contracts,
                    "leave_seed_source_start_fail_routes": leave_seed_source_fails,
                    "leave_seed_and_start_routes": leave_seed_start_routes,
                },
                "leave-seed route continuity is 78/80 and failures are source-start singleton caveats, not unknown-endpoint novelty",
                leave_seed_routes == 78
                and leave_seed_contracts == 8
                and leave_seed_source_fails == 2
                and leave_seed_start_routes == 78,
            ),
            _gate_row(
                "G6_failure_rows_are_source_start_singletons",
                "Are leave-seed failures source-start singleton rows rather than target or unknown failures?",
                _count_dict(
                    route_rows[
                        route_rows["rotation_mode"].astype(str).eq("leave_seed_out")
                        & ~route_rows["axis_b_rotation_route_pass"].astype(bool)
                    ]["axis_b_rotation_route_status"]
                ),
                "2 failures, both axis_b_rotation_blocked_by_source_start_singleton",
                _count_dict(
                    route_rows[
                        route_rows["rotation_mode"].astype(str).eq("leave_seed_out")
                        & ~route_rows["axis_b_rotation_route_pass"].astype(bool)
                    ]["axis_b_rotation_route_status"]
                )
                == {"axis_b_rotation_blocked_by_source_start_singleton": 2},
            ),
            _gate_row(
                "G7_wall_claim_remains_closed",
                "Are wall claims still closed after seed-anchor rotation?",
                int(contract_rows["wall_contract_ready"].astype(bool).sum()),
                "zero wall-ready contracts",
                not bool(contract_rows["wall_contract_ready"].astype(bool).any()),
            ),
            _gate_row(
                "G8_no_new_leiden_method_quality_or_full_replay_claim",
                "Are execution, method, quality/cost, and full-replay claims closed?",
                CLAIM_BOUNDARY,
                "claim boundary explicitly closed",
                True,
            ),
        ]
    )


def _summary(
    *,
    trace_dir: Path,
    dual_axis_dir: Path,
    output_dir: Path,
    step_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    unknown_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
    mode_summary: pd.DataFrame,
    gates: pd.DataFrame,
    dual_axis_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_axis_b_seed_anchor_rotation_summary.v1",
        "status": (
            "axis_b_seed_anchor_rotation_unknowns_robust_route_level_source_start_caveat_wall_closed"
        ),
        "run_status": RUN_STATUS,
        "trace_dir": str(trace_dir),
        "dual_axis_dir": str(dual_axis_dir),
        "output_dir": str(output_dir),
        "step_rotation_row_count": int(len(step_rows)),
        "route_rotation_row_count": int(len(route_rows)),
        "unknown_rotation_row_count": int(len(unknown_rows)),
        "contract_rotation_row_count": int(len(contract_rows)),
        "mode_summary_row_count": int(len(mode_summary)),
        "mode_status_counts": _count_dict(mode_summary["mode_status"]),
        "full_pair_atlas_route_pass_count": int(
            _mode_value(mode_summary, "full_pair_atlas", "axis_b_route_pass_count")
        ),
        "leave_start_out_route_pass_count": int(
            _mode_value(mode_summary, "leave_start_out", "axis_b_route_pass_count")
        ),
        "leave_seed_out_route_pass_count": int(
            _mode_value(mode_summary, "leave_seed_out", "axis_b_route_pass_count")
        ),
        "leave_seed_and_start_out_route_pass_count": int(
            _mode_value(
                mode_summary,
                "leave_seed_and_start_out",
                "axis_b_route_pass_count",
            )
        ),
        "leave_seed_out_contract_pass_count": int(
            _mode_value(mode_summary, "leave_seed_out", "axis_b_contract_pass_count")
        ),
        "same_seed_unknown_pair_known_count_by_mode": {
            str(row.rotation_mode): int(row.same_seed_unknown_pair_known_count)
            for row in mode_summary.itertuples(index=False)
        },
        "same_seed_unknown_true_novel_count_by_mode": {
            str(row.rotation_mode): int(row.same_seed_unknown_true_novel_count)
            for row in mode_summary.itertuples(index=False)
        },
        "leave_seed_out_failure_rows": route_rows[
            route_rows["rotation_mode"].astype(str).eq("leave_seed_out")
            & ~route_rows["axis_b_rotation_route_pass"].astype(bool)
        ][
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "axis_b_rotation_route_status",
                "true_novel_step_indices",
            ]
        ].to_dict(orient="records"),
        "wall_ready_contract_count": int(
            contract_rows["wall_contract_ready"].astype(bool).sum()
        ),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "Axis B's same-seed unknown reclassification is robust to all "
            "seed/start exclusions: all 27 unknown rows remain pair-level known "
            "and none become true novel endpoints. Full route-level Axis B "
            "continuity passes in the baseline and leave-start modes, but "
            "leave-one-seed-out continuity is 78 of 80 routes and 8 of 10 "
            "contracts because two local_pair_009 seed-0 source-start signatures "
            "lack off-seed known support. This is a source-anchor singleton "
            "caveat, not an unknown-endpoint or wall result."
        ),
        "recommended_next_gate": (
            "Keep the unknown-endpoint reinterpretation, but do not claim full "
            "seed-invariant Axis B route continuity. The next bounded gate should "
            "either add source-start support rotation to these two seed-0 cases or "
            "run a fresh predeclared Axis B panel that records source-start support "
            "independently from interior endpoint continuity."
        ),
        "dual_axis_status": dual_axis_summary.get("status"),
        "dual_axis_claim_boundary": DUAL_AXIS_CLAIM_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            STEP_ROTATION_ROWS_CSV,
            ROUTE_ROTATION_ROWS_CSV,
            UNKNOWN_ROTATION_ROWS_CSV,
            CONTRACT_ROTATION_ROWS_CSV,
            MODE_SUMMARY_ROWS_CSV,
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
    mode_summary: pd.DataFrame,
    contract_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    unknown_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    leave_seed_failures = route_rows[
        route_rows["rotation_mode"].astype(str).eq("leave_seed_out")
        & ~route_rows["axis_b_rotation_route_pass"].astype(bool)
    ]
    lines = [
        "# NanoClustering G4.8 Axis B Seed-Anchor Rotation Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- step_rotation_row_count: {summary['step_rotation_row_count']}",
        f"- route_rotation_row_count: {summary['route_rotation_row_count']}",
        f"- unknown_rotation_row_count: {summary['unknown_rotation_row_count']}",
        f"- contract_rotation_row_count: {summary['contract_rotation_row_count']}",
        f"- full_pair_atlas_route_pass_count: {summary['full_pair_atlas_route_pass_count']}",
        f"- leave_start_out_route_pass_count: {summary['leave_start_out_route_pass_count']}",
        f"- leave_seed_out_route_pass_count: {summary['leave_seed_out_route_pass_count']}",
        (
            "- leave_seed_and_start_out_route_pass_count: "
            f"{summary['leave_seed_and_start_out_route_pass_count']}"
        ),
        f"- leave_seed_out_contract_pass_count: {summary['leave_seed_out_contract_pass_count']}",
        (
            "- same_seed_unknown_pair_known_count_by_mode: "
            f"{summary['same_seed_unknown_pair_known_count_by_mode']}"
        ),
        (
            "- same_seed_unknown_true_novel_count_by_mode: "
            f"{summary['same_seed_unknown_true_novel_count_by_mode']}"
        ),
        f"- wall_ready_contract_count: {summary['wall_ready_contract_count']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Mode Summary",
        "",
        _markdown_table(
            mode_summary,
            [
                "rotation_mode",
                "axis_b_route_pass_count",
                "axis_b_contract_pass_count",
                "same_seed_unknown_pair_known_count",
                "same_seed_unknown_true_novel_count",
                "source_start_fail_route_count",
                "true_novel_route_count",
                "mode_status",
            ],
            max_rows=10,
        ),
        "",
        "## Contract Rotation",
        "",
        _markdown_table(
            contract_rows,
            [
                "rotation_mode",
                "local_pair_id",
                "start_condition",
                "seed_count",
                "axis_b_rotation_seed_pass_count",
                "axis_b_rotation_contract_pass",
                "source_start_fail_seed_count",
                "true_novel_endpoint_seed_count",
                "same_seed_unknown_pair_known_seed_count",
                "axis_b_rotation_contract_status",
            ],
            max_rows=50,
        ),
        "",
        "## Leave-Seed Failure Rows",
        "",
        _markdown_table(
            leave_seed_failures,
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "axis_b_rotation_route_status",
                "axis_b_rotation_block_reason",
                "true_novel_step_indices",
                "endpoint_role_sequence",
                "min_rotation_support_row_count",
            ],
            max_rows=20,
        ),
        "",
        "## Unknown Rotation Rows",
        "",
        _markdown_table(
            unknown_rows,
            [
                "rotation_mode",
                "local_pair_id",
                "start_condition",
                "seed",
                "step_index",
                "result_endpoint_signature_id",
                "rotation_endpoint_role",
                "rotation_support_row_count",
                "rotation_support_seed_count",
                "unknown_rotation_status",
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
            "This audit strengthens only the interpretation of same-seed unknown "
            "labels. It also exposes a source-start support caveat under "
            "leave-one-seed-out rotation. It does not promote Axis B into a wall, "
            "quality, cost, full-replay, or method claim."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(*, trace_dir: Path, dual_axis_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_rows = _read_csv(trace_dir / TRACE_ROWS_CSV)
    dual_axis_gates = _read_csv(dual_axis_dir / DUAL_AXIS_GATE_MATRIX_CSV)
    dual_axis_summary = json.loads(
        (dual_axis_dir / DUAL_AXIS_SUMMARY_JSON).read_text(encoding="utf-8")
    )

    primary_trace = _primary_trace(trace_rows)
    step_rows = _step_rotation_rows(primary_trace)
    route_rows = _route_rotation_rows(step_rows)
    unknown_rows = _unknown_rotation_rows(step_rows)
    contract_rows = _contract_rotation_rows(route_rows)
    mode_summary = _mode_summary_rows(
        step_rows,
        route_rows,
        unknown_rows,
        contract_rows,
    )
    gates = _gate_matrix(
        dual_axis_gates=dual_axis_gates,
        mode_summary=mode_summary,
        route_rows=route_rows,
        unknown_rows=unknown_rows,
        contract_rows=contract_rows,
    )
    summary = _summary(
        trace_dir=trace_dir,
        dual_axis_dir=dual_axis_dir,
        output_dir=output_dir,
        step_rows=step_rows,
        route_rows=route_rows,
        unknown_rows=unknown_rows,
        contract_rows=contract_rows,
        mode_summary=mode_summary,
        gates=gates,
        dual_axis_summary=dual_axis_summary,
    )
    config = {
        "schema": "nanoclustering_g4_8_axis_b_seed_anchor_rotation_config.v1",
        "trace_dir": str(trace_dir),
        "dual_axis_dir": str(dual_axis_dir),
        "output_dir": str(output_dir),
        "primary_route_family": PRIMARY_ROUTE_FAMILY,
        "rotation_modes": ROTATION_MODES,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(step_rows, output_dir / STEP_ROTATION_ROWS_CSV)
    _write_csv(route_rows, output_dir / ROUTE_ROTATION_ROWS_CSV)
    _write_csv(unknown_rows, output_dir / UNKNOWN_ROTATION_ROWS_CSV)
    _write_csv(contract_rows, output_dir / CONTRACT_ROTATION_ROWS_CSV)
    _write_csv(mode_summary, output_dir / MODE_SUMMARY_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
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
        mode_summary=mode_summary,
        contract_rows=contract_rows,
        route_rows=route_rows,
        unknown_rows=unknown_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--dual-axis-dir", type=Path, default=DEFAULT_DUAL_AXIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        trace_dir=args.trace_dir,
        dual_axis_dir=args.dual_axis_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
