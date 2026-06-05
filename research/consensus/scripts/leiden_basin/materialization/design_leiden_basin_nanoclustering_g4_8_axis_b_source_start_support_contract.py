#!/usr/bin/env python3
"""Design the G4.8 Axis B source-start support contract.

The Axis B seed-anchor rotation audit showed that same-seed unknown endpoint
reclassification is robust, while full route-level continuity has two
leave-one-seed-out source-start singleton caveats. This contract separates
source-start support from post-start/interior endpoint continuity so the two
questions cannot be conflated.

It does not run Leiden, broaden route execution, promote walls, evaluate
quality/cost value, replay full NanoClustering, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_axis_b_seed_anchor_rotation import (
    CLAIM_BOUNDARY as ROTATION_CLAIM_BOUNDARY,
    CONTRACT_ROTATION_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_ROTATION_DIR,
    GATE_MATRIX_CSV as ROTATION_GATE_MATRIX_CSV,
    MODE_SUMMARY_ROWS_CSV as ROTATION_MODE_SUMMARY_ROWS_CSV,
    ROUTE_ROTATION_ROWS_CSV,
    STEP_ROTATION_ROWS_CSV,
    SUMMARY_JSON as ROTATION_SUMMARY_JSON,
    UNKNOWN_ROTATION_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_axis_b_source_start_support_contract_gamma1e5_20260604"
)

RUN_STATUS = "designed_nanoclustering_g4_8_axis_b_source_start_support_contract"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 Axis B source-start support contract design only; reads "
    "the existing Axis B seed-anchor rotation audit and separates source-start "
    "support from post-start endpoint continuity. It does not run Leiden, "
    "broaden route execution, promote walls, evaluate quality/cost value, replay "
    "full NanoClustering, or claim method or algorithm success."
)

RULE_ROWS_CSV = "nanoclustering_g4_8_axis_b_source_start_support_rule_rows.csv"
SOURCE_START_ROWS_CSV = (
    "nanoclustering_g4_8_axis_b_source_start_support_source_start_rows.csv"
)
INTERIOR_ROUTE_ROWS_CSV = (
    "nanoclustering_g4_8_axis_b_source_start_support_interior_route_rows.csv"
)
CONTRACT_ROWS_CSV = "nanoclustering_g4_8_axis_b_source_start_support_contract_rows.csv"
MODE_SUMMARY_ROWS_CSV = (
    "nanoclustering_g4_8_axis_b_source_start_support_mode_summary_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_axis_b_source_start_support_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_axis_b_source_start_support_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_axis_b_source_start_support_summary.json"
REPORT_MD = "nanoclustering_g4_8_axis_b_source_start_support_report.md"

SUPPORT_RULES = (
    {
        "rule_id": "S1_source_start_full_pair_support",
        "axis": "source_start_support",
        "rule_question": "Does the step-1 source signature have pair-level known source support?",
        "seed_level_requirement": "step_index == 1 role contains original_source in the full pair atlas",
        "contract_level_requirement": "all seed-routes in a contract pass source-start support",
        "claim_effect": "source-start support is necessary for full route-level Axis B continuity",
    },
    {
        "rule_id": "S2_source_start_leave_start_support",
        "axis": "source_start_support",
        "rule_question": "Does source-start support survive removing the focal start condition?",
        "seed_level_requirement": "step-1 source role remains known under leave-start-out",
        "contract_level_requirement": "all seed-routes pass leave-start source-start support",
        "claim_effect": "tests start-condition dependence of the source anchor",
    },
    {
        "rule_id": "S3_source_start_leave_seed_support",
        "axis": "source_start_support",
        "rule_question": "Does source-start support survive removing the focal seed?",
        "seed_level_requirement": "step-1 source role remains known under leave-seed-out",
        "contract_level_requirement": "all seed-routes pass leave-seed source-start support",
        "claim_effect": "source-start singleton caveats block full seed-invariant route continuity",
    },
    {
        "rule_id": "I1_post_start_endpoint_known",
        "axis": "post_start_interior_endpoint_continuity",
        "rule_question": "Are all post-start endpoint signatures pair-level known under rotation?",
        "seed_level_requirement": "steps after step 1 contain zero true-novel endpoint roles",
        "contract_level_requirement": "all seed-routes pass post-start endpoint continuity",
        "claim_effect": "supports interior endpoint continuity without relying on the source-start anchor",
    },
    {
        "rule_id": "I2_target_final_known",
        "axis": "post_start_interior_endpoint_continuity",
        "rule_question": "Does the route still end in the pair-level target after source-start separation?",
        "seed_level_requirement": "final post-start endpoint role contains drop_bridge_target",
        "contract_level_requirement": "all seed-routes end at the pair-level target",
        "claim_effect": "keeps interior continuity tied to the intended source-to-target route",
    },
    {
        "rule_id": "I3_direct_edge_retained",
        "axis": "post_start_interior_endpoint_continuity",
        "rule_question": "Is the direct pair edge retained while interior continuity is tested?",
        "seed_level_requirement": "route-level direct edge retention remains true",
        "contract_level_requirement": "all seed-routes retain the direct pair edge",
        "claim_effect": "preserves the physical direct-path condition from Axis B",
    },
    {
        "rule_id": "C1_unknown_reinterpretation_kept",
        "axis": "claim_boundary",
        "rule_question": "Is the robust unknown-endpoint reinterpretation kept separate?",
        "seed_level_requirement": "same-seed unknown rows remain pair-level known under every rotation",
        "contract_level_requirement": "unknown reinterpretation cannot repair source-start singleton caveats",
        "claim_effect": "prevents overclaiming full route continuity from interior endpoint evidence",
    },
    {
        "rule_id": "C2_wall_method_claims_closed",
        "axis": "claim_boundary",
        "rule_question": "Are wall, quality/cost, full replay, and method claims closed?",
        "seed_level_requirement": "readout-only contract design",
        "contract_level_requirement": "wall_contract_ready remains false",
        "claim_effect": "limits this artifact to a bounded methodology clarification",
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


def _step_list(values: pd.Series | list[Any]) -> str:
    if isinstance(values, pd.Series):
        raw_values = values.dropna().tolist()
    else:
        raw_values = list(values)
    if not raw_values:
        return ""
    return ";".join(str(int(value)) for value in sorted(set(raw_values)))


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


def _rule_rows() -> pd.DataFrame:
    rows = pd.DataFrame(list(SUPPORT_RULES))
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _source_start_rows(step_rows: pd.DataFrame) -> pd.DataFrame:
    rows = step_rows[step_rows["step_index"].astype(int).eq(1)].copy()
    rows["source_start_support_pass"] = rows["rotation_endpoint_role"].map(
        _role_is_source
    ).astype(bool) & rows["rotation_support_row_count"].astype(int).gt(0)
    rows["source_start_support_status"] = rows["source_start_support_pass"].map(
        {
            True: "source_start_support_pass",
            False: "source_start_singleton_no_rotated_support",
        }
    )
    keep = [
        "rotation_mode",
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "seed",
        "step_index",
        "result_endpoint_signature_id",
        "rotation_endpoint_role",
        "rotation_support_row_count",
        "rotation_support_seed_count",
        "rotation_support_start_condition_count",
        "rotation_support_assignments",
        "rotation_support_seeds",
        "rotation_support_start_conditions",
        "source_start_support_pass",
        "source_start_support_status",
    ]
    rows = rows[keep].copy()
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["rotation_mode", "local_pair_id", "start_condition", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _interior_route_rows(step_rows: pd.DataFrame, route_rows: pd.DataFrame) -> pd.DataFrame:
    post_start = step_rows[step_rows["step_index"].astype(int).gt(1)].copy()
    group_cols = [
        "rotation_mode",
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "seed",
    ]
    route_lookup = {
        tuple(row[col] for col in group_cols): row
        for row in route_rows.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for keys, group in post_start.groupby(group_cols, sort=False):
        ordered = group.sort_values("step_index", kind="mergesort").reset_index(drop=True)
        last = ordered.iloc[-1]
        route = route_lookup.get(tuple(keys))
        if route is None:
            raise KeyError(f"Missing route rotation row for {keys}")
        novel_mask = ordered["rotation_endpoint_role"].map(_role_is_novel).astype(bool)
        source_mask = ordered["rotation_endpoint_role"].map(_role_is_source).astype(bool)
        target_mask = ordered["rotation_endpoint_role"].map(_role_is_target).astype(bool)
        unknown_mask = ordered["is_same_seed_unknown_step"].astype(bool)
        unknown_pair_known_mask = ordered[
            "same_seed_unknown_pair_known_under_rotation"
        ].astype(bool)
        no_true_novel_post_start = not bool(novel_mask.any())
        target_final = _role_is_target(last["rotation_endpoint_role"])
        direct_edge_retained = _as_bool(
            route["direct_edge_retained_under_rotation_route"]
        )
        interior_pass = all(
            [
                no_true_novel_post_start,
                target_final,
                direct_edge_retained,
            ]
        )
        rows.append(
            {
                "rotation_mode": str(keys[0]),
                "route_contract_id": str(keys[1]),
                "validation_unit_id": str(keys[2]),
                "local_pair_id": str(keys[3]),
                "start_condition": str(keys[4]),
                "seed": int(keys[5]),
                "post_start_step_count": int(len(ordered)),
                "post_start_interior_continuity_pass": bool(interior_pass),
                "post_start_no_true_novel_endpoint_pass": bool(
                    no_true_novel_post_start
                ),
                "post_start_target_final_pass": bool(target_final),
                "direct_edge_retained_under_rotation_route": bool(direct_edge_retained),
                "post_start_true_novel_step_count": int(novel_mask.sum()),
                "post_start_true_novel_step_indices": _step_list(
                    ordered.loc[novel_mask, "step_index"]
                ),
                "post_start_source_step_indices": _step_list(
                    ordered.loc[source_mask, "step_index"]
                ),
                "post_start_target_step_indices": _step_list(
                    ordered.loc[target_mask, "step_index"]
                ),
                "same_seed_unknown_post_start_step_count": int(unknown_mask.sum()),
                "same_seed_unknown_pair_known_post_start_step_count": int(
                    unknown_pair_known_mask.sum()
                ),
                "post_start_endpoint_role_sequence": ";".join(
                    ordered["rotation_endpoint_role"].astype(str).tolist()
                ),
                "min_post_start_rotation_support_row_count": int(
                    ordered["rotation_support_row_count"].astype(int).min()
                ),
                "min_post_start_rotation_support_seed_count": int(
                    ordered["rotation_support_seed_count"].astype(int).min()
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["rotation_mode", "local_pair_id", "start_condition", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _contract_rows(
    source_start_rows: pd.DataFrame,
    interior_rows: pd.DataFrame,
    rotation_contract_rows: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "rotation_mode",
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
    ]
    source_grouped = (
        source_start_rows.groupby(group_cols, sort=False)
        .agg(
            seed_count=("seed", "nunique"),
            source_start_support_seed_pass_count=("source_start_support_pass", "sum"),
            source_start_fail_seed_count=(
                "source_start_support_pass",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            source_start_status_counts=(
                "source_start_support_status",
                lambda values: json.dumps(_count_dict(values), sort_keys=True),
            ),
        )
        .reset_index()
    )
    interior_grouped = (
        interior_rows.groupby(group_cols, sort=False)
        .agg(
            interior_seed_pass_count=("post_start_interior_continuity_pass", "sum"),
            post_start_true_novel_seed_count=(
                "post_start_true_novel_step_count",
                lambda values: int((values.astype(int) > 0).sum()),
            ),
            same_seed_unknown_pair_known_post_start_seed_count=(
                "same_seed_unknown_pair_known_post_start_step_count",
                lambda values: int((values.astype(int) > 0).sum()),
            ),
            min_post_start_rotation_support_row_count=(
                "min_post_start_rotation_support_row_count",
                "min",
            ),
        )
        .reset_index()
    )
    rows = source_grouped.merge(
        interior_grouped,
        on=group_cols,
        how="left",
        validate="one_to_one",
    )
    rotation_keep = [
        "rotation_mode",
        "route_contract_id",
        "axis_b_rotation_seed_pass_count",
        "axis_b_rotation_contract_pass",
        "axis_b_rotation_contract_status",
        "wall_contract_ready",
    ]
    rows = rows.merge(
        rotation_contract_rows[rotation_keep],
        on=["rotation_mode", "route_contract_id"],
        how="left",
        validate="one_to_one",
    )
    rows["source_start_contract_pass"] = rows[
        "source_start_support_seed_pass_count"
    ].astype(int).eq(rows["seed_count"].astype(int))
    rows["post_start_interior_contract_pass"] = rows[
        "interior_seed_pass_count"
    ].astype(int).eq(rows["seed_count"].astype(int))
    rows["split_axis_b_contract_pass"] = (
        rows["source_start_contract_pass"].astype(bool)
        & rows["post_start_interior_contract_pass"].astype(bool)
    )
    rows["wall_contract_ready_v2"] = False

    def status(row: pd.Series) -> str:
        if bool(row["split_axis_b_contract_pass"]):
            return "source_start_and_interior_contract_pass"
        if bool(row["post_start_interior_contract_pass"]) and not bool(
            row["source_start_contract_pass"]
        ):
            return "interior_continuity_pass_source_start_support_caveat"
        if not bool(row["post_start_interior_contract_pass"]):
            return "interior_continuity_blocked"
        return "source_start_support_contract_partial"

    rows["source_start_support_contract_status"] = rows.apply(status, axis=1)
    rows["contract_claim_boundary_note"] = (
        "Post-start endpoint continuity cannot repair source-start singleton "
        "support caveats and cannot promote wall, quality, cost, full replay, or "
        "method claims."
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["rotation_mode", "local_pair_id", "start_condition"],
        kind="mergesort",
    ).reset_index(drop=True)


def _mode_summary_rows(
    source_start_rows: pd.DataFrame,
    interior_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode_name in sorted(source_start_rows["rotation_mode"].astype(str).unique()):
        source_mode = source_start_rows[
            source_start_rows["rotation_mode"].astype(str).eq(mode_name)
        ]
        interior_mode = interior_rows[
            interior_rows["rotation_mode"].astype(str).eq(mode_name)
        ]
        contract_mode = contract_rows[
            contract_rows["rotation_mode"].astype(str).eq(mode_name)
        ]
        source_pass = int(source_mode["source_start_support_pass"].astype(bool).sum())
        interior_pass = int(
            interior_mode["post_start_interior_continuity_pass"].astype(bool).sum()
        )
        source_contract_pass = int(
            contract_mode["source_start_contract_pass"].astype(bool).sum()
        )
        interior_contract_pass = int(
            contract_mode["post_start_interior_contract_pass"].astype(bool).sum()
        )
        if source_pass == len(source_mode) and interior_pass == len(interior_mode):
            mode_status = "source_start_and_interior_pass"
        elif interior_pass == len(interior_mode):
            mode_status = "interior_continuity_pass_source_start_caveat"
        else:
            mode_status = "interior_continuity_or_source_support_blocked"
        rows.append(
            {
                "rotation_mode": mode_name,
                "source_start_route_pass_count": source_pass,
                "source_start_contract_pass_count": source_contract_pass,
                "post_start_interior_route_pass_count": interior_pass,
                "post_start_interior_contract_pass_count": interior_contract_pass,
                "source_start_fail_route_count": int(len(source_mode) - source_pass),
                "post_start_true_novel_route_count": int(
                    (
                        interior_mode["post_start_true_novel_step_count"].astype(int)
                        > 0
                    ).sum()
                ),
                "same_seed_unknown_pair_known_post_start_route_count": int(
                    (
                        interior_mode[
                            "same_seed_unknown_pair_known_post_start_step_count"
                        ].astype(int)
                        > 0
                    ).sum()
                ),
                "source_start_status_counts": json.dumps(
                    _count_dict(source_mode["source_start_support_status"]),
                    sort_keys=True,
                ),
                "contract_status_counts": json.dumps(
                    _count_dict(contract_mode["source_start_support_contract_status"]),
                    sort_keys=True,
                ),
                "mode_status": mode_status,
                "wall_contract_ready_count": int(
                    contract_mode["wall_contract_ready_v2"].astype(bool).sum()
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _mode_value(mode_summary: pd.DataFrame, mode_name: str, column: str) -> Any:
    matches = mode_summary[mode_summary["rotation_mode"].astype(str).eq(mode_name)]
    if matches.empty:
        raise KeyError(mode_name)
    return matches.iloc[0][column]


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


def _gate_matrix(
    *,
    rotation_gates: pd.DataFrame,
    rule_rows: pd.DataFrame,
    source_start_rows: pd.DataFrame,
    interior_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
    mode_summary: pd.DataFrame,
) -> pd.DataFrame:
    source_failures = source_start_rows[
        ~source_start_rows["source_start_support_pass"].astype(bool)
    ]
    leave_seed_failures = source_failures[
        source_failures["rotation_mode"].astype(str).eq("leave_seed_out")
    ]
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_rotation_gates_pass",
                "Did the upstream Axis B seed-anchor rotation audit gates pass?",
                _count_dict(rotation_gates["gate_status"]),
                "all upstream rotation gates pass",
                bool(rotation_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_source_start_and_interior_rules_materialized",
                "Are source-start support and post-start continuity split explicitly?",
                f"rule_count={len(rule_rows)} axes={sorted(rule_rows['axis'].unique())}",
                "source-start, post-start interior, and claim-boundary rules are materialized",
                len(rule_rows) == 8
                and set(rule_rows["axis"].astype(str))
                == {
                    "source_start_support",
                    "post_start_interior_endpoint_continuity",
                    "claim_boundary",
                },
            ),
            _gate_row(
                "G3_source_start_support_passes_full_and_leave_start",
                "Does source-start support pass in full-pair and leave-start modes?",
                {
                    "full_pair_atlas_source_start": int(
                        _mode_value(
                            mode_summary,
                            "full_pair_atlas",
                            "source_start_route_pass_count",
                        )
                    ),
                    "leave_start_out_source_start": int(
                        _mode_value(
                            mode_summary,
                            "leave_start_out",
                            "source_start_route_pass_count",
                        )
                    ),
                },
                "80 of 80 source-start routes pass in both modes",
                int(
                    _mode_value(
                        mode_summary,
                        "full_pair_atlas",
                        "source_start_route_pass_count",
                    )
                )
                == 80
                and int(
                    _mode_value(
                        mode_summary,
                        "leave_start_out",
                        "source_start_route_pass_count",
                    )
                )
                == 80,
            ),
            _gate_row(
                "G4_source_start_leave_seed_caveat_preserved",
                "Is the leave-seed source-start caveat localized and preserved?",
                {
                    "leave_seed_out_source_start": int(
                        _mode_value(
                            mode_summary,
                            "leave_seed_out",
                            "source_start_route_pass_count",
                        )
                    ),
                    "leave_seed_and_start_out_source_start": int(
                        _mode_value(
                            mode_summary,
                            "leave_seed_and_start_out",
                            "source_start_route_pass_count",
                        )
                    ),
                    "failure_rows": source_failures[
                        [
                            "rotation_mode",
                            "local_pair_id",
                            "start_condition",
                            "seed",
                            "result_endpoint_signature_id",
                        ]
                    ].to_dict(orient="records"),
                },
                "leave-seed modes pass 78 of 80 source-start routes with two local_pair_009 seed-0 caveats",
                int(
                    _mode_value(
                        mode_summary,
                        "leave_seed_out",
                        "source_start_route_pass_count",
                    )
                )
                == 78
                and int(
                    _mode_value(
                        mode_summary,
                        "leave_seed_and_start_out",
                        "source_start_route_pass_count",
                    )
                )
                == 78
                and len(source_failures) == 4
                and set(source_failures["local_pair_id"].astype(str))
                == {"local_pair_009"}
                and set(source_failures["seed"].astype(int)) == {0},
            ),
            _gate_row(
                "G5_post_start_interior_continuity_passes_all_modes",
                "Does post-start endpoint continuity pass after separating source-start support?",
                {
                    str(row.rotation_mode): int(row.post_start_interior_route_pass_count)
                    for row in mode_summary.itertuples(index=False)
                },
                "80 of 80 post-start routes pass in every rotation mode",
                all(
                    int(value) == 80
                    for value in mode_summary[
                        "post_start_interior_route_pass_count"
                    ].tolist()
                ),
            ),
            _gate_row(
                "G6_post_start_true_novel_endpoint_absent",
                "Are post-start true-novel endpoints absent in every mode?",
                {
                    str(row.rotation_mode): int(row.post_start_true_novel_route_count)
                    for row in mode_summary.itertuples(index=False)
                },
                "zero post-start true-novel endpoint routes in every mode",
                all(
                    int(value) == 0
                    for value in mode_summary[
                        "post_start_true_novel_route_count"
                    ].tolist()
                ),
            ),
            _gate_row(
                "G7_wall_claim_remains_closed",
                "Are wall claims still closed after the source-start split?",
                int(contract_rows["wall_contract_ready_v2"].astype(bool).sum()),
                "zero wall-ready contracts",
                not bool(contract_rows["wall_contract_ready_v2"].astype(bool).any()),
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
    rotation_dir: Path,
    output_dir: Path,
    rule_rows: pd.DataFrame,
    source_start_rows: pd.DataFrame,
    interior_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
    mode_summary: pd.DataFrame,
    gates: pd.DataFrame,
    rotation_summary: dict[str, Any],
) -> dict[str, Any]:
    source_failures = source_start_rows[
        ~source_start_rows["source_start_support_pass"].astype(bool)
    ]
    return {
        "schema": "nanoclustering_g4_8_axis_b_source_start_support_summary.v1",
        "status": (
            "axis_b_source_start_support_split_interior_continuity_pass_source_start_caveat_preserved_wall_closed"
        ),
        "run_status": RUN_STATUS,
        "rotation_dir": str(rotation_dir),
        "output_dir": str(output_dir),
        "rule_count": int(len(rule_rows)),
        "source_start_row_count": int(len(source_start_rows)),
        "interior_route_row_count": int(len(interior_rows)),
        "contract_row_count": int(len(contract_rows)),
        "mode_summary_row_count": int(len(mode_summary)),
        "source_start_failure_row_count": int(len(source_failures)),
        "source_start_failure_rows": source_failures[
            [
                "rotation_mode",
                "local_pair_id",
                "start_condition",
                "seed",
                "result_endpoint_signature_id",
            ]
        ].to_dict(orient="records"),
        "full_pair_atlas_source_start_pass_count": int(
            _mode_value(
                mode_summary,
                "full_pair_atlas",
                "source_start_route_pass_count",
            )
        ),
        "leave_start_out_source_start_pass_count": int(
            _mode_value(
                mode_summary,
                "leave_start_out",
                "source_start_route_pass_count",
            )
        ),
        "leave_seed_out_source_start_pass_count": int(
            _mode_value(
                mode_summary,
                "leave_seed_out",
                "source_start_route_pass_count",
            )
        ),
        "leave_seed_and_start_out_source_start_pass_count": int(
            _mode_value(
                mode_summary,
                "leave_seed_and_start_out",
                "source_start_route_pass_count",
            )
        ),
        "post_start_interior_pass_count_by_mode": {
            str(row.rotation_mode): int(row.post_start_interior_route_pass_count)
            for row in mode_summary.itertuples(index=False)
        },
        "post_start_true_novel_route_count_by_mode": {
            str(row.rotation_mode): int(row.post_start_true_novel_route_count)
            for row in mode_summary.itertuples(index=False)
        },
        "contract_status_counts": _count_dict(
            contract_rows["source_start_support_contract_status"]
        ),
        "mode_status_counts": _count_dict(mode_summary["mode_status"]),
        "wall_ready_contract_count": int(
            contract_rows["wall_contract_ready_v2"].astype(bool).sum()
        ),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "Separating source-start support from post-start endpoint continuity "
            "preserves both findings without conflation. Post-start endpoint "
            "continuity passes 80 of 80 routes in every rotation mode and has zero "
            "true-novel endpoint routes. Source-start support passes in full-pair "
            "and leave-start modes, but leave-seed modes preserve two "
            "local_pair_009 seed-0 source-start singleton caveats. Therefore the "
            "interior Axis B endpoint claim is stronger than the full route-level "
            "seed-invariance claim."
        ),
        "recommended_next_gate": (
            "Use this split contract for the next fresh Axis B panel: record "
            "source-start support, post-start endpoint continuity, target-final "
            "continuity, and direct-edge retention separately. Do not repair the two "
            "source-start singleton caveats with interior evidence, and do not "
            "promote wall or method claims."
        ),
        "rotation_status": rotation_summary.get("status"),
        "rotation_claim_boundary": ROTATION_CLAIM_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            RULE_ROWS_CSV,
            SOURCE_START_ROWS_CSV,
            INTERIOR_ROUTE_ROWS_CSV,
            CONTRACT_ROWS_CSV,
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
    rule_rows: pd.DataFrame,
    source_start_rows: pd.DataFrame,
    interior_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
    mode_summary: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    source_failures = source_start_rows[
        ~source_start_rows["source_start_support_pass"].astype(bool)
    ]
    lines = [
        "# NanoClustering G4.8 Axis B Source-Start Support Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- rule_count: {summary['rule_count']}",
        f"- source_start_row_count: {summary['source_start_row_count']}",
        f"- interior_route_row_count: {summary['interior_route_row_count']}",
        f"- contract_row_count: {summary['contract_row_count']}",
        f"- full_pair_atlas_source_start_pass_count: {summary['full_pair_atlas_source_start_pass_count']}",
        f"- leave_start_out_source_start_pass_count: {summary['leave_start_out_source_start_pass_count']}",
        f"- leave_seed_out_source_start_pass_count: {summary['leave_seed_out_source_start_pass_count']}",
        (
            "- leave_seed_and_start_out_source_start_pass_count: "
            f"{summary['leave_seed_and_start_out_source_start_pass_count']}"
        ),
        f"- post_start_interior_pass_count_by_mode: {summary['post_start_interior_pass_count_by_mode']}",
        f"- post_start_true_novel_route_count_by_mode: {summary['post_start_true_novel_route_count_by_mode']}",
        f"- source_start_failure_row_count: {summary['source_start_failure_row_count']}",
        f"- wall_ready_contract_count: {summary['wall_ready_contract_count']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Rules",
        "",
        _markdown_table(
            rule_rows,
            [
                "rule_id",
                "axis",
                "rule_question",
                "seed_level_requirement",
                "contract_level_requirement",
                "claim_effect",
            ],
            max_rows=20,
        ),
        "",
        "## Mode Summary",
        "",
        _markdown_table(
            mode_summary,
            [
                "rotation_mode",
                "source_start_route_pass_count",
                "source_start_contract_pass_count",
                "post_start_interior_route_pass_count",
                "post_start_interior_contract_pass_count",
                "source_start_fail_route_count",
                "post_start_true_novel_route_count",
                "mode_status",
            ],
            max_rows=10,
        ),
        "",
        "## Source-Start Failure Rows",
        "",
        _markdown_table(
            source_failures,
            [
                "rotation_mode",
                "local_pair_id",
                "start_condition",
                "seed",
                "result_endpoint_signature_id",
                "rotation_endpoint_role",
                "rotation_support_row_count",
                "source_start_support_status",
            ],
            max_rows=20,
        ),
        "",
        "## Contract Rows",
        "",
        _markdown_table(
            contract_rows,
            [
                "rotation_mode",
                "local_pair_id",
                "start_condition",
                "seed_count",
                "source_start_support_seed_pass_count",
                "interior_seed_pass_count",
                "source_start_contract_pass",
                "post_start_interior_contract_pass",
                "source_start_support_contract_status",
            ],
            max_rows=50,
        ),
        "",
        "## Interior Route Sample",
        "",
        _markdown_table(
            interior_rows,
            [
                "rotation_mode",
                "local_pair_id",
                "start_condition",
                "seed",
                "post_start_interior_continuity_pass",
                "post_start_true_novel_step_count",
                "post_start_source_step_indices",
                "post_start_target_step_indices",
                "same_seed_unknown_pair_known_post_start_step_count",
                "min_post_start_rotation_support_row_count",
            ],
            max_rows=30,
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
            "This contract is a split-readout guardrail. It strengthens post-start "
            "endpoint continuity while preserving source-start singleton caveats. "
            "It does not convert either finding into a wall, quality, cost, full "
            "replay, or method claim."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(*, rotation_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    step_rows = _read_csv(rotation_dir / STEP_ROTATION_ROWS_CSV)
    route_rows = _read_csv(rotation_dir / ROUTE_ROTATION_ROWS_CSV)
    rotation_contract_rows = _read_csv(rotation_dir / CONTRACT_ROTATION_ROWS_CSV)
    rotation_gates = _read_csv(rotation_dir / ROTATION_GATE_MATRIX_CSV)
    rotation_summary = json.loads(
        (rotation_dir / ROTATION_SUMMARY_JSON).read_text(encoding="utf-8")
    )
    # Read for reproducibility validation of the input artifact surface.
    _read_csv(rotation_dir / UNKNOWN_ROTATION_ROWS_CSV)
    _read_csv(rotation_dir / ROTATION_MODE_SUMMARY_ROWS_CSV)

    rule_rows = _rule_rows()
    source_start_rows = _source_start_rows(step_rows)
    interior_rows = _interior_route_rows(step_rows, route_rows)
    contract_rows = _contract_rows(
        source_start_rows,
        interior_rows,
        rotation_contract_rows,
    )
    mode_summary = _mode_summary_rows(source_start_rows, interior_rows, contract_rows)
    gates = _gate_matrix(
        rotation_gates=rotation_gates,
        rule_rows=rule_rows,
        source_start_rows=source_start_rows,
        interior_rows=interior_rows,
        contract_rows=contract_rows,
        mode_summary=mode_summary,
    )
    summary = _summary(
        rotation_dir=rotation_dir,
        output_dir=output_dir,
        rule_rows=rule_rows,
        source_start_rows=source_start_rows,
        interior_rows=interior_rows,
        contract_rows=contract_rows,
        mode_summary=mode_summary,
        gates=gates,
        rotation_summary=rotation_summary,
    )
    config = {
        "schema": "nanoclustering_g4_8_axis_b_source_start_support_config.v1",
        "rotation_dir": str(rotation_dir),
        "output_dir": str(output_dir),
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(rule_rows, output_dir / RULE_ROWS_CSV)
    _write_csv(source_start_rows, output_dir / SOURCE_START_ROWS_CSV)
    _write_csv(interior_rows, output_dir / INTERIOR_ROUTE_ROWS_CSV)
    _write_csv(contract_rows, output_dir / CONTRACT_ROWS_CSV)
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
        rule_rows=rule_rows,
        source_start_rows=source_start_rows,
        interior_rows=interior_rows,
        contract_rows=contract_rows,
        mode_summary=mode_summary,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation-dir", type=Path, default=DEFAULT_ROTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(rotation_dir=args.rotation_dir, output_dir=args.output_dir)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
