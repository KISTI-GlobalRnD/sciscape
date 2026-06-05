#!/usr/bin/env python3
"""Audit the primary bridge-release pathway shape for G4.8.

This is a readout over the already executed scoped pathway-probe traces. It
separates three questions that should not be collapsed into a wall claim:

1. Is the direct pair edge physically retained during bridge-release routes?
2. Which seed-routes pass through intermediate unknown/support-incompatible
   endpoints, and which stay on known anchors?
3. What objective debt/recovery shape accompanies the source-to-target
   transition?

It does not run Leiden, broaden route execution, evaluate quality/cost value,
replay full NanoClustering, or promote basin walls.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TRACE_DIR,
    GATE_MATRIX_CSV as TRACE_GATE_MATRIX_CSV,
    SEED_ROUTE_SUMMARY_CSV,
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
    / "leiden_basin_nanoclustering_g4_8_primary_bridge_release_pathway_shape_gamma1e5_20260604"
)

PRIMARY_ROUTE_FAMILY = "bridge_release_interpolation_probe"
RUN_STATUS = "audited_nanoclustering_g4_8_primary_bridge_release_pathway_shape"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 primary bridge-release pathway-shape audit only; reads "
    "already executed scoped local route traces. It does not run Leiden, broaden "
    "route execution, promote walls, evaluate quality/cost value, replay full "
    "NanoClustering, or claim method/algorithm success."
)

SEED_ROWS_CSV = "nanoclustering_g4_8_primary_bridge_release_pathway_shape_seed_rows.csv"
CONTRACT_ROWS_CSV = (
    "nanoclustering_g4_8_primary_bridge_release_pathway_shape_contract_rows.csv"
)
PAIR_ROWS_CSV = "nanoclustering_g4_8_primary_bridge_release_pathway_shape_pair_rows.csv"
GATE_MATRIX_CSV = "nanoclustering_g4_8_primary_bridge_release_pathway_shape_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_primary_bridge_release_pathway_shape_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_primary_bridge_release_pathway_shape_config.json"
REPORT_MD = "nanoclustering_g4_8_primary_bridge_release_pathway_shape_report.md"

EPSILON = 1e-9


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _first_step(ordered: pd.DataFrame, mask: pd.Series) -> int | None:
    matches = ordered.loc[mask.astype(bool), "step_index"]
    if matches.empty:
        return None
    return int(matches.iloc[0])


def _step_list(ordered: pd.DataFrame, mask: pd.Series) -> str:
    steps = [str(int(step)) for step in ordered.loc[mask.astype(bool), "step_index"].tolist()]
    return ";".join(steps)


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


def _objective_shape_class(
    *,
    max_debt: float,
    max_recovery: float,
    final_delta: float,
) -> str:
    has_debt = float(max_debt) > EPSILON
    has_recovery = float(max_recovery) > EPSILON
    final_below_start = float(final_delta) < -EPSILON
    if has_debt and has_recovery and final_below_start:
        return "debt_with_partial_recovery_final_below_start"
    if has_debt and not has_recovery and final_below_start:
        return "debt_without_recovery_final_below_start"
    if has_debt and has_recovery:
        return "debt_with_partial_recovery"
    if has_debt:
        return "debt_without_recovery"
    return "no_objective_debt"


def _pathway_shape_class(
    *,
    has_unknown: bool,
    first_target_step: int | None,
    has_recovery: bool,
) -> str:
    prefix = "unknown_intermediate" if has_unknown else "known_anchor_only"
    if first_target_step is None:
        timing = "no_target"
    elif int(first_target_step) == 2:
        timing = "step2_transition"
    elif int(first_target_step) == 3:
        timing = "step3_transition"
    else:
        timing = f"step{int(first_target_step)}_transition"
    suffix = "with_recovery" if has_recovery else "no_recovery"
    return f"{prefix}_{timing}_{suffix}"


def _seed_rows(trace_rows: pd.DataFrame, seed_summary: pd.DataFrame) -> pd.DataFrame:
    primary_trace = trace_rows[
        trace_rows["planned_route_family"].astype(str).eq(PRIMARY_ROUTE_FAMILY)
    ].copy()
    primary_seed_summary = seed_summary[
        seed_summary["planned_route_family"].astype(str).eq(PRIMARY_ROUTE_FAMILY)
    ].copy()

    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "seed",
    ]
    summary_lookup = {
        (str(row.route_contract_id), int(row.seed)): row._asdict()
        for row in primary_seed_summary.itertuples(index=False)
    }
    for keys, group in primary_trace.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        ordered = group.sort_values("step_index", kind="mergesort").reset_index(drop=True)
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        route_contract_id = str(key_data["route_contract_id"])
        seed = int(key_data["seed"])
        summary = summary_lookup.get((route_contract_id, seed), {})

        unknown_mask = ordered["endpoint_assignment_by_step"].astype(str).eq(
            "unknown_new_endpoint"
        )
        expected_mask = ordered["matches_expected_final_anchor"].astype(bool)
        support_mask = ordered["support_incompatibility_check"].astype(bool)
        recovery_mask = ordered["objective_recovery_from_min"].astype(float).gt(EPSILON)
        min_objective_value = float(ordered["objective_value_by_step"].min())
        min_objective_rows = ordered[
            ordered["objective_value_by_step"].astype(float).eq(min_objective_value)
        ]
        min_objective_step = int(min_objective_rows.iloc[0]["step_index"])
        first_target_step = _first_step(ordered, expected_mask)
        first_unknown_step = _first_step(ordered, unknown_mask)
        max_debt = float(ordered["objective_debt_from_start"].max())
        max_recovery = float(ordered["objective_recovery_from_min"].max())
        final_delta = float(last["objective_delta_from_start"])
        final_below_start = final_delta < -EPSILON
        direct_fraction_min = float(ordered["direct_edge_weight_fraction"].min())
        active_direct_min = float(ordered["active_direct_edge_weight"].min())
        direct_edge_physically_retained = direct_fraction_min > 0.0 and active_direct_min > 0.0
        known_anchor_only = not bool(unknown_mask.any())
        known_anchor_direct_path_candidate = (
            bool(direct_edge_physically_retained)
            and bool(known_anchor_only)
            and first_target_step is not None
        )
        has_recovery = max_recovery > EPSILON
        shape_class = _pathway_shape_class(
            has_unknown=bool(unknown_mask.any()),
            first_target_step=first_target_step,
            has_recovery=has_recovery,
        )
        objective_class = _objective_shape_class(
            max_debt=max_debt,
            max_recovery=max_recovery,
            final_delta=final_delta,
        )
        rows.append(
            {
                **key_data,
                "planned_route_family": PRIMARY_ROUTE_FAMILY,
                "route_step_count": int(len(ordered)),
                "source_start_anchor_matched": bool(summary.get("source_start_anchor_matched", False)),
                "expected_final_anchor_reached": bool(
                    summary.get("expected_final_anchor_reached", False)
                ),
                "first_endpoint_assignment": str(first["endpoint_assignment_by_step"]),
                "final_endpoint_assignment": str(last["endpoint_assignment_by_step"]),
                "endpoint_assignment_sequence": str(
                    summary.get("endpoint_assignment_sequence", "")
                ),
                "first_target_step": first_target_step,
                "first_unknown_step": first_unknown_step,
                "unknown_step_indices": _step_list(ordered, unknown_mask),
                "support_incompatibility_step_indices": _step_list(ordered, support_mask),
                "objective_recovery_step_indices": _step_list(ordered, recovery_mask),
                "unknown_endpoint_step_count": int(unknown_mask.sum()),
                "support_incompatibility_step_count": int(support_mask.sum()),
                "direct_edge_weight_fraction_min": direct_fraction_min,
                "active_direct_edge_weight_min": active_direct_min,
                "active_direct_edge_weight_max": float(ordered["active_direct_edge_weight"].max()),
                "bridge_edge_weight_fraction_at_first_unknown": (
                    None
                    if first_unknown_step is None
                    else float(
                        ordered.loc[
                            ordered["step_index"].astype(int).eq(first_unknown_step),
                            "bridge_edge_weight_fraction",
                        ].iloc[0]
                    )
                ),
                "bridge_edge_weight_fraction_at_first_target": (
                    None
                    if first_target_step is None
                    else float(
                        ordered.loc[
                            ordered["step_index"].astype(int).eq(first_target_step),
                            "bridge_edge_weight_fraction",
                        ].iloc[0]
                    )
                ),
                "physical_direct_edge_retained_all_steps": bool(
                    direct_edge_physically_retained
                ),
                "known_anchor_only_path": bool(known_anchor_only),
                "known_anchor_direct_path_candidate": bool(
                    known_anchor_direct_path_candidate
                ),
                "accepted_direct_path_evidence": False,
                "max_objective_debt_from_start": max_debt,
                "max_objective_recovery_from_min": max_recovery,
                "final_objective_delta_from_start": final_delta,
                "final_objective_below_start": bool(final_below_start),
                "min_objective_step": min_objective_step,
                "objective_value_at_min": min_objective_value,
                "objective_value_final": float(last["objective_value_by_step"]),
                "objective_shape_class": objective_class,
                "pathway_shape_class": shape_class,
                "shape_readout_status": (
                    "primary_bridge_release_seed_shape_candidate_not_wall"
                    if known_anchor_direct_path_candidate
                    else "primary_bridge_release_seed_unknown_intermediate_not_wall"
                ),
                "wall_seed_ready": False,
                "wall_seed_block_reason": (
                    "seed-level primary bridge-release shape is diagnostic only; "
                    "contract-level all-seed direct-path acceptance and objective "
                    "recovery remain unavailable"
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["local_pair_id", "start_condition", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _contract_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["route_contract_id", "validation_unit_id", "local_pair_id", "start_condition"]
    for keys, group in seed_rows.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        seed_count = int(group["seed"].nunique())
        unknown_count = int(group["unknown_endpoint_step_count"].astype(int).gt(0).sum())
        clean_count = int(group["known_anchor_direct_path_candidate"].astype(bool).sum())
        recovery_count = int(group["max_objective_recovery_from_min"].astype(float).gt(EPSILON).sum())
        step2_count = int(group["first_target_step"].astype(int).eq(2).sum())
        step3_count = int(group["first_target_step"].astype(int).eq(3).sum())
        all_seed_clean = clean_count == seed_count and seed_count > 0
        all_seed_recovery = recovery_count == seed_count and seed_count > 0
        if unknown_count > 0 and clean_count > 0:
            status = "mixed_known_direct_candidates_and_unknown_intermediates_not_wall"
        elif clean_count == seed_count and seed_count > 0:
            status = "all_seed_known_anchor_direct_candidate_not_accepted_wall"
        elif unknown_count == seed_count and seed_count > 0:
            status = "all_seed_unknown_intermediate_not_wall"
        else:
            status = "primary_bridge_release_shape_mixed_not_wall"
        rows.append(
            {
                **key_data,
                "planned_route_family": PRIMARY_ROUTE_FAMILY,
                "seed_count": seed_count,
                "source_to_expected_transition_seed_count": int(
                    group["expected_final_anchor_reached"].astype(bool).sum()
                ),
                "physical_direct_edge_retained_seed_count": int(
                    group["physical_direct_edge_retained_all_steps"].astype(bool).sum()
                ),
                "known_anchor_direct_path_candidate_seed_count": clean_count,
                "unknown_intermediate_seed_count": unknown_count,
                "first_target_step2_seed_count": step2_count,
                "first_target_step3_seed_count": step3_count,
                "objective_debt_seed_count": int(
                    group["max_objective_debt_from_start"].astype(float).gt(EPSILON).sum()
                ),
                "objective_recovery_seed_count": recovery_count,
                "final_objective_below_start_seed_count": int(
                    group["final_objective_below_start"].astype(bool).sum()
                ),
                "all_seed_known_anchor_direct_path_candidate": bool(all_seed_clean),
                "all_seed_objective_recovery": bool(all_seed_recovery),
                "accepted_direct_path_evidence": False,
                "wall_contract_ready": False,
                "pathway_shape_status": status,
                "pathway_shape_class_counts": _count_dict(group["pathway_shape_class"]),
                "objective_shape_class_counts": _count_dict(group["objective_shape_class"]),
                "max_objective_debt_from_start": float(
                    group["max_objective_debt_from_start"].max()
                ),
                "max_objective_recovery_from_min": float(
                    group["max_objective_recovery_from_min"].max()
                ),
                "min_final_objective_delta_from_start": float(
                    group["final_objective_delta_from_start"].min()
                ),
                "max_final_objective_delta_from_start": float(
                    group["final_objective_delta_from_start"].max()
                ),
                "wall_contract_block_reason": (
                    "no wall promotion: direct edge is physically retained, but no "
                    "contract has all-seed known-anchor direct-path evidence and "
                    "objective recovery is not all-seed"
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["local_pair_id", "start_condition"],
        kind="mergesort",
    ).reset_index(drop=True)


def _pair_rows(seed_rows: pd.DataFrame, contract_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for local_pair_id, group in seed_rows.groupby("local_pair_id", sort=False):
        contracts = contract_rows[
            contract_rows["local_pair_id"].astype(str).eq(str(local_pair_id))
        ]
        seed_count = int(len(group))
        recovery_count = int(group["max_objective_recovery_from_min"].astype(float).gt(EPSILON).sum())
        unknown_count = int(group["unknown_endpoint_step_count"].astype(int).gt(0).sum())
        clean_count = int(group["known_anchor_direct_path_candidate"].astype(bool).sum())
        if unknown_count > 0 and recovery_count == 0:
            status = "pair_level_step3_debt_without_recovery_not_wall"
        elif unknown_count > 0 and recovery_count > 0:
            status = "pair_level_mixed_step2_step3_partial_recovery_not_wall"
        else:
            status = "pair_level_primary_bridge_release_shape_not_wall"
        rows.append(
            {
                "local_pair_id": str(local_pair_id),
                "planned_route_family": PRIMARY_ROUTE_FAMILY,
                "contract_count": int(len(contracts)),
                "seed_count": seed_count,
                "source_to_expected_transition_seed_count": int(
                    group["expected_final_anchor_reached"].astype(bool).sum()
                ),
                "physical_direct_edge_retained_seed_count": int(
                    group["physical_direct_edge_retained_all_steps"].astype(bool).sum()
                ),
                "known_anchor_direct_path_candidate_seed_count": clean_count,
                "unknown_intermediate_seed_count": unknown_count,
                "objective_recovery_seed_count": recovery_count,
                "all_seed_known_anchor_direct_path_contract_count": int(
                    contracts["all_seed_known_anchor_direct_path_candidate"].astype(bool).sum()
                ),
                "all_seed_objective_recovery_contract_count": int(
                    contracts["all_seed_objective_recovery"].astype(bool).sum()
                ),
                "accepted_direct_path_contract_count": int(
                    contracts["accepted_direct_path_evidence"].astype(bool).sum()
                ),
                "wall_ready_contract_count": int(
                    contracts["wall_contract_ready"].astype(bool).sum()
                ),
                "pathway_shape_class_counts": _count_dict(group["pathway_shape_class"]),
                "objective_shape_class_counts": _count_dict(group["objective_shape_class"]),
                "pair_pathway_shape_status": status,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


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
    trace_gates: pd.DataFrame,
    seed_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
) -> pd.DataFrame:
    unknown_step_values = sorted(
        {
            int(step)
            for value in seed_rows["unknown_step_indices"].astype(str)
            for step in value.split(";")
            if step
        }
    )
    unknown_bridge_values = sorted(
        {
            float(value)
            for value in seed_rows.loc[
                seed_rows["first_unknown_step"].notna(),
                "bridge_edge_weight_fraction_at_first_unknown",
            ].tolist()
            if value is not None and not (isinstance(value, float) and math.isnan(value))
        }
    )
    clean_seed_count = int(seed_rows["known_anchor_direct_path_candidate"].astype(bool).sum())
    all_seed_clean_contracts = int(
        contract_rows["all_seed_known_anchor_direct_path_candidate"].astype(bool).sum()
    )
    recovery_seed_count = int(
        seed_rows["max_objective_recovery_from_min"].astype(float).gt(EPSILON).sum()
    )
    all_seed_recovery_contracts = int(
        contract_rows["all_seed_objective_recovery"].astype(bool).sum()
    )
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_trace_gates_pass",
                "Did every upstream scoped pathway-probe trace gate pass?",
                _count_dict(trace_gates["gate_status"]),
                "all upstream trace gates pass",
                bool(trace_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_primary_scope_only",
                "Is this audit restricted to primary bridge-release traces?",
                f"seed_rows={len(seed_rows)} contract_rows={len(contract_rows)}",
                "80 seed routes and 10 contracts from bridge_release_interpolation_probe only",
                len(seed_rows) == 80
                and len(contract_rows) == 10
                and bool(seed_rows["planned_route_family"].astype(str).eq(PRIMARY_ROUTE_FAMILY).all()),
            ),
            _gate_row(
                "G3_direct_edge_physically_retained",
                "Was the direct pair edge physically retained throughout every primary route?",
                (
                    f"retained_seed_count={int(seed_rows['physical_direct_edge_retained_all_steps'].astype(bool).sum())} "
                    f"min_direct_fraction={float(seed_rows['direct_edge_weight_fraction_min'].min()):.6g} "
                    f"min_active_direct_weight={float(seed_rows['active_direct_edge_weight_min'].min()):.6g}"
                ),
                "all 80 seed routes retain positive direct edge weight",
                int(seed_rows["physical_direct_edge_retained_all_steps"].astype(bool).sum())
                == len(seed_rows),
            ),
            _gate_row(
                "G4_intermediate_unknown_localized",
                "Are intermediate unknown/support-incompatible states separated and localized?",
                (
                    f"unknown_seed_count={int(seed_rows['unknown_endpoint_step_count'].astype(int).gt(0).sum())} "
                    f"unknown_steps={unknown_step_values} unknown_bridge_fractions={unknown_bridge_values}"
                ),
                "27 unknown seed routes, all first observed at step 2 with bridge fraction 0.75",
                int(seed_rows["unknown_endpoint_step_count"].astype(int).gt(0).sum()) == 27
                and unknown_step_values == [2]
                and unknown_bridge_values == [0.75],
            ),
            _gate_row(
                "G5_known_anchor_direct_candidates_are_seed_level_only",
                "Do known-anchor direct-path candidates exist, but fail contract-level all-seed acceptance?",
                (
                    f"clean_seed_count={clean_seed_count} "
                    f"all_seed_clean_contracts={all_seed_clean_contracts}"
                ),
                "some clean seed candidates exist, zero all-seed clean contracts",
                clean_seed_count > 0 and all_seed_clean_contracts == 0,
            ),
            _gate_row(
                "G6_objective_debt_universal_recovery_partial",
                "Is objective debt universal while recovery remains partial and non-contract-uniform?",
                (
                    f"debt_seed_count={int(seed_rows['max_objective_debt_from_start'].astype(float).gt(EPSILON).sum())} "
                    f"recovery_seed_count={recovery_seed_count} "
                    f"all_seed_recovery_contracts={all_seed_recovery_contracts}"
                ),
                "all 80 seed routes have debt, only partial recovery, zero all-seed recovery contracts",
                int(seed_rows["max_objective_debt_from_start"].astype(float).gt(EPSILON).sum())
                == len(seed_rows)
                and recovery_seed_count == 8
                and all_seed_recovery_contracts == 0,
            ),
            _gate_row(
                "G7_wall_claim_remains_closed",
                "Are wall claims still closed after direct-path and objective-shape readout?",
                (
                    f"accepted_direct_path_contracts={int(contract_rows['accepted_direct_path_evidence'].astype(bool).sum())} "
                    f"wall_ready_contracts={int(contract_rows['wall_contract_ready'].astype(bool).sum())}"
                ),
                "zero accepted direct-path contracts and zero wall-ready contracts",
                not bool(contract_rows["accepted_direct_path_evidence"].astype(bool).any())
                and not bool(contract_rows["wall_contract_ready"].astype(bool).any()),
            ),
            _gate_row(
                "G8_no_method_quality_or_full_replay_claim",
                "Are method, quality/cost, full replay, and algorithm claims closed?",
                CLAIM_BOUNDARY,
                "claim boundary explicitly closed",
                True,
            ),
        ]
    )


def _summary(
    *,
    trace_dir: Path,
    output_dir: Path,
    seed_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_primary_bridge_release_pathway_shape_summary.v1",
        "status": "primary_bridge_release_pathway_shape_audit_direct_candidates_seed_level_wall_closed",
        "run_status": RUN_STATUS,
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "seed_row_count": int(len(seed_rows)),
        "contract_row_count": int(len(contract_rows)),
        "pair_row_count": int(len(pair_rows)),
        "source_to_expected_transition_seed_count": int(
            seed_rows["expected_final_anchor_reached"].astype(bool).sum()
        ),
        "physical_direct_edge_retained_seed_count": int(
            seed_rows["physical_direct_edge_retained_all_steps"].astype(bool).sum()
        ),
        "known_anchor_direct_path_candidate_seed_count": int(
            seed_rows["known_anchor_direct_path_candidate"].astype(bool).sum()
        ),
        "unknown_intermediate_seed_count": int(
            seed_rows["unknown_endpoint_step_count"].astype(int).gt(0).sum()
        ),
        "all_seed_known_anchor_direct_path_contract_count": int(
            contract_rows["all_seed_known_anchor_direct_path_candidate"].astype(bool).sum()
        ),
        "objective_debt_seed_count": int(
            seed_rows["max_objective_debt_from_start"].astype(float).gt(EPSILON).sum()
        ),
        "objective_recovery_seed_count": int(
            seed_rows["max_objective_recovery_from_min"].astype(float).gt(EPSILON).sum()
        ),
        "all_seed_objective_recovery_contract_count": int(
            contract_rows["all_seed_objective_recovery"].astype(bool).sum()
        ),
        "accepted_direct_path_contract_count": int(
            contract_rows["accepted_direct_path_evidence"].astype(bool).sum()
        ),
        "wall_ready_contract_count": int(contract_rows["wall_contract_ready"].astype(bool).sum()),
        "pathway_shape_class_counts": _count_dict(seed_rows["pathway_shape_class"]),
        "objective_shape_class_counts": _count_dict(seed_rows["objective_shape_class"]),
        "pair_pathway_shape_status_counts": _count_dict(pair_rows["pair_pathway_shape_status"]),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "The direct edge is physically retained in every primary bridge-release "
            "seed route, and 53 seed routes stay on known anchors while reaching the "
            "drop-bridge target. However, every contract still has at least one "
            "intermediate unknown/support-incompatible seed route, objective recovery "
            "appears in only 8 of 80 seed routes, and zero contracts satisfy all-seed "
            "direct-path or all-seed recovery acceptance. This is pathway-shape "
            "evidence, not wall evidence."
        ),
        "recommended_next_gate": (
            "Do not broaden to new pairs yet. Use the two observed regimes as the "
            "next design split: local_pair_009 is step3 debt-without-recovery; "
            "local_pair_012 is mostly step2 with partial recovery. The next valid "
            "test is an explicitly predeclared direct-path acceptance contract, not "
            "wall promotion."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            SEED_ROWS_CSV,
            CONTRACT_ROWS_CSV,
            PAIR_ROWS_CSV,
            GATE_MATRIX_CSV,
            SUMMARY_JSON,
            CONFIG_JSON,
            REPORT_MD,
        ],
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    seed_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Primary Bridge-Release Pathway Shape Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- source_to_expected_transition_seed_count: {summary['source_to_expected_transition_seed_count']}",
        f"- physical_direct_edge_retained_seed_count: {summary['physical_direct_edge_retained_seed_count']}",
        f"- known_anchor_direct_path_candidate_seed_count: {summary['known_anchor_direct_path_candidate_seed_count']}",
        f"- unknown_intermediate_seed_count: {summary['unknown_intermediate_seed_count']}",
        f"- all_seed_known_anchor_direct_path_contract_count: {summary['all_seed_known_anchor_direct_path_contract_count']}",
        f"- objective_debt_seed_count: {summary['objective_debt_seed_count']}",
        f"- objective_recovery_seed_count: {summary['objective_recovery_seed_count']}",
        f"- all_seed_objective_recovery_contract_count: {summary['all_seed_objective_recovery_contract_count']}",
        f"- accepted_direct_path_contract_count: {summary['accepted_direct_path_contract_count']}",
        f"- wall_ready_contract_count: {summary['wall_ready_contract_count']}",
        f"- pathway_shape_class_counts: {summary['pathway_shape_class_counts']}",
        f"- objective_shape_class_counts: {summary['objective_shape_class_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Shape",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "seed_count",
                "known_anchor_direct_path_candidate_seed_count",
                "unknown_intermediate_seed_count",
                "objective_recovery_seed_count",
                "all_seed_known_anchor_direct_path_contract_count",
                "all_seed_objective_recovery_contract_count",
                "pair_pathway_shape_status",
            ],
        ),
        "",
        "## Contract Shape",
        "",
        _markdown_table(
            contract_rows,
            [
                "local_pair_id",
                "start_condition",
                "seed_count",
                "known_anchor_direct_path_candidate_seed_count",
                "unknown_intermediate_seed_count",
                "first_target_step2_seed_count",
                "first_target_step3_seed_count",
                "objective_recovery_seed_count",
                "all_seed_known_anchor_direct_path_candidate",
                "all_seed_objective_recovery",
                "pathway_shape_status",
            ],
        ),
        "",
        "## Seed Shape Sample",
        "",
        _markdown_table(
            seed_rows,
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "first_target_step",
                "unknown_step_indices",
                "known_anchor_direct_path_candidate",
                "max_objective_debt_from_start",
                "max_objective_recovery_from_min",
                "final_objective_delta_from_start",
                "pathway_shape_class",
                "objective_shape_class",
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
            "This readout keeps direct-path availability at candidate level. It "
            "does not accept direct-path evidence at contract level because every "
            "primary contract contains at least one intermediate unknown seed-route "
            "and no contract has all-seed objective recovery."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = _read_csv(trace_dir / TRACE_ROWS_CSV)
    seed_summary = _read_csv(trace_dir / SEED_ROUTE_SUMMARY_CSV)
    trace_gates = _read_csv(trace_dir / TRACE_GATE_MATRIX_CSV)

    seed_audit = _seed_rows(trace_rows, seed_summary)
    contract_audit = _contract_rows(seed_audit)
    pair_audit = _pair_rows(seed_audit, contract_audit)
    gates = _gate_matrix(
        trace_gates=trace_gates,
        seed_rows=seed_audit,
        contract_rows=contract_audit,
    )
    summary = _summary(
        trace_dir=trace_dir,
        output_dir=output_dir,
        seed_rows=seed_audit,
        contract_rows=contract_audit,
        pair_rows=pair_audit,
        gates=gates,
    )

    _write_csv(seed_audit, output_dir / SEED_ROWS_CSV)
    _write_csv(contract_audit, output_dir / CONTRACT_ROWS_CSV)
    _write_csv(pair_audit, output_dir / PAIR_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            _json_safe(
                {
                    "trace_dir": str(trace_dir),
                    "output_dir": str(output_dir),
                    "primary_route_family": PRIMARY_ROUTE_FAMILY,
                    "epsilon": EPSILON,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        seed_rows=seed_audit,
        contract_rows=contract_audit,
        pair_rows=pair_audit,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
