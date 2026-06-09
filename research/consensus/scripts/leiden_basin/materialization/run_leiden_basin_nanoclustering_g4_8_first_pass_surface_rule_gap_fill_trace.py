#!/usr/bin/env python3
"""Run the first-pass surface-rule gap-fill trace contract.

This runner consumes
``design_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract.py``.
It executes exactly the contract's fraction-expanded route rows for ``001`` and
``007`` on the existing local induced graph surface, then groups those rows by
pair/start/seed to classify diagnostic recurrence, scoreable-negative behavior,
or residual gaps.

The run is a local readout only. It does not run full NanoClustering, promote
wall/pathway labels, evaluate quality/cost value, replay full NanoClustering,
or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from design_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract import (
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTRACT_DIR,
    GAP_FILL_CANDIDATE_IDS,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    ROUTE_PLAN_ROWS_CSV as CONTRACT_ROUTE_PLAN_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace import (
    _edge_weight_parts,
    _parse_node_ids,
    _scaled_local_edges,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from run_leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_LOCAL_ABLATION_DIR,
    LOCAL_GRAPH_ROWS_CSV,
    _build_igraph,
    _canonical_groups,
    _collect_induced_edges_by_branch,
    _initial_membership,
    _mechanism_read,
    _node_doc_lookup,
    _read_json,
    _signature_id,
)
from sciscape.clustering.runner import LeidenRunner


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_gamma1e5_20260609"
)

TRACE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_rows.csv"
)
SEED_ROUTE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_seed_route_rows.csv"
)
PAIR_READOUT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_pair_rows.csv"
)
FRACTION_READOUT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_fraction_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_report.md"

RUN_STATUS = "executed_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace"
ROUTE_EXECUTION_STATUS = "executed_surface_rule_gap_fill_local_fraction_trace"
WALL_PROMOTION_STATUS = "not_promoted_gap_fill_trace_only"
METHOD_STATUS = "surface_rule_gap_fill_trace_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass surface-rule gap-fill trace only; executes "
    "the predeclared 001/007 local bridge-fraction readout. It does not run full "
    "NanoClustering, promote wall/pathway labels, evaluate quality/cost value, "
    "replay full NanoClustering, or claim method success."
)

SOURCE_FAMILY_MECHANISMS = {
    "pair_coassigned_with_selected_bridge",
    "pair_separated_bridge_split",
}
SINGLE_SIDE_MECHANISM = "pair_separated_single_side_bridge"
TARGET_LIKE_MECHANISM = "pair_coassigned_without_selected_bridge"
EXPECTED_ROUTE_PLAN_ROWS = 54
EPS = 1e-9


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _count_dict(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.astype(str).value_counts(dropna=False).sort_index().items()
    }


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
        "observed": json.dumps(_json_safe(observed), sort_keys=True),
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(max_rows)
    if visible.empty:
        return "_No rows._"

    def cell(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(_json_safe(value), sort_keys=True).replace("|", "\\|")
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


def _trace_rows(
    *,
    route_plan: pd.DataFrame,
    contract_dir: Path,
    local_ablation_dir: Path,
    gamma: float,
    seeds: int,
    n_iterations: int,
    edge_chunk_size: int,
) -> tuple[pd.DataFrame, int]:
    local_config = _read_json(
        local_ablation_dir
        / "nanoclustering_symmetric_object_variable_pair_local_ablation_config.json"
    )
    graph_mechanism_dir = Path(str(local_config["graph_mechanism_dir"]))
    difference_dir = Path(str(local_config["difference_dir"]))
    node_rows = _read_csv(
        difference_dir / "nanoclustering_symmetric_object_terminal_difference_node_rows.csv"
    )
    doc_lookup = _node_doc_lookup(node_rows)
    local_specs = _read_csv(local_ablation_dir / LOCAL_GRAPH_ROWS_CSV)
    candidate_pair_ids = set(route_plan["local_pair_id"].astype(str))
    specs = local_specs[local_specs["local_pair_id"].astype(str).isin(candidate_pair_ids)].copy()
    if specs["local_pair_id"].nunique() != len(candidate_pair_ids):
        missing = sorted(candidate_pair_ids - set(specs["local_pair_id"].astype(str)))
        raise ValueError(f"missing local specs for candidate pairs: {missing}")

    target_nodes_by_branch: dict[str, set[int]] = {}
    for spec in specs.itertuples(index=False):
        target_nodes_by_branch.setdefault(str(spec.branch), set()).update(
            _parse_node_ids(spec.local_node_ids)
        )
    induced_edges_by_branch = _collect_induced_edges_by_branch(
        graph_mechanism_dir=graph_mechanism_dir,
        target_nodes_by_branch=target_nodes_by_branch,
        edge_chunk_size=int(edge_chunk_size),
    )
    spec_by_pair = {
        str(row.local_pair_id): row._asdict()
        for row in specs.sort_values("local_pair_id", kind="mergesort").itertuples(index=False)
    }

    output_rows: list[dict[str, Any]] = []
    ordered_plan = route_plan.sort_values(
        ["local_pair_id", "start_condition", "fraction_order"],
        kind="mergesort",
    )
    for route in ordered_plan.itertuples(index=False):
        route_data = route._asdict()
        local_pair_id = str(route_data["local_pair_id"])
        spec = spec_by_pair[local_pair_id]
        object_role_id = str(spec["object_role_universe_id"])
        branch = str(spec["branch"])
        left = int(spec["left_node_id"])
        right = int(spec["right_node_id"])
        node_ids = _parse_node_ids(spec["local_node_ids"])
        bridge_nodes = set(_parse_node_ids(spec["selected_bridge_node_ids"]))
        node_sizes = [
            int(doc_lookup.get((object_role_id, int(node_id)), 1))
            for node_id in node_ids
        ]
        induced_edges = induced_edges_by_branch.get(
            branch,
            pd.DataFrame(columns=["source", "target", "weight"]),
        )
        bridge_fraction = float(route_data["bridge_fraction"])
        local_edges = _scaled_local_edges(
            induced_edges=induced_edges,
            node_ids=node_ids,
            left_node=left,
            right_node=right,
            bridge_nodes=bridge_nodes,
            direct_fraction=1.0,
            bridge_fraction=bridge_fraction,
        )
        edge_parts = _edge_weight_parts(
            induced_edges=induced_edges,
            node_ids=node_ids,
            left_node=left,
            right_node=right,
            bridge_nodes=bridge_nodes,
        )
        graph = _build_igraph(node_ids, local_edges)
        runner = LeidenRunner(graph, objective="cpm", default_iterations=int(n_iterations))
        initial_membership = _initial_membership(
            start_condition=str(route_data["start_condition"]),
            node_ids=node_ids,
            left_node=left,
            right_node=right,
            bridge_nodes=bridge_nodes,
        )
        initial_signature = _signature_id(_canonical_groups(node_ids, initial_membership))
        for seed in range(int(seeds)):
            result = runner.run(
                float(gamma),
                seed=int(seed),
                initial_membership=initial_membership,
                node_sizes=node_sizes,
            )
            membership = list(map(int, result.membership))
            groups = _canonical_groups(node_ids, membership)
            result_signature_id = _signature_id(groups)
            read = _mechanism_read(
                membership=membership,
                node_ids=node_ids,
                left_node=left,
                right_node=right,
                bridge_nodes=bridge_nodes,
            )
            output_rows.append(
                {
                    "trace_row_id": (
                        f"{route_data['route_contract_id']}__seed{seed:02d}"
                    ),
                    "route_contract_id": str(route_data["route_contract_id"]),
                    "route_sequence_id": (
                        f"{local_pair_id}__{route_data['start_condition']}__seed{seed:02d}"
                    ),
                    "local_pair_id": local_pair_id,
                    "next_contract_role": str(route_data["next_contract_role"]),
                    "route_family": str(route_data["route_family"]),
                    "start_condition": str(route_data["start_condition"]),
                    "start_condition_macro_role": str(
                        route_data["start_condition_macro_role"]
                    ),
                    "start_condition_expected_validation_pass": _as_bool(
                        route_data["start_condition_expected_validation_pass"]
                    ),
                    "fraction_order": int(route_data["fraction_order"]),
                    "seed": int(seed),
                    "gamma": float(gamma),
                    "n_iterations": int(n_iterations),
                    "direct_edge_weight_fraction": 1.0,
                    "bridge_edge_weight_fraction": bridge_fraction,
                    "local_node_count": int(len(node_ids)),
                    "selected_bridge_count": int(len(bridge_nodes)),
                    "local_edge_count": int(graph.ecount()),
                    "local_edge_weight_sum": (
                        float(sum(graph.es["weight"])) if graph.ecount() else 0.0
                    ),
                    "active_direct_edge_weight": float(
                        edge_parts["original_direct_edge_weight"]
                    ),
                    "active_pair_bridge_edge_weight_sum": float(
                        edge_parts["original_pair_bridge_edge_weight_sum"]
                        * bridge_fraction
                    ),
                    **edge_parts,
                    "initial_endpoint_signature_id": initial_signature,
                    "result_endpoint_signature_id": result_signature_id,
                    "result_endpoint_signature": json.dumps(groups, sort_keys=True),
                    "objective_value_by_fraction": float(result.quality),
                    "cluster_count": int(result.cluster_count),
                    **read,
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "run_status": RUN_STATUS,
                    "contract_dir": str(contract_dir),
                    "local_ablation_dir": str(local_ablation_dir),
                }
            )
    rows = pd.DataFrame(output_rows)
    if rows.empty:
        return rows, int(specs["local_pair_id"].nunique())
    rows = rows.sort_values(
        ["local_pair_id", "start_condition", "seed", "fraction_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    group_cols = ["local_pair_id", "start_condition", "seed"]
    rows["objective_start_value"] = rows.groupby(group_cols, sort=False)[
        "objective_value_by_fraction"
    ].transform("first")
    rows["objective_delta_from_start"] = (
        rows["objective_value_by_fraction"] - rows["objective_start_value"]
    )
    rows["objective_debt_from_start"] = np.maximum(
        0.0,
        rows["objective_start_value"] - rows["objective_value_by_fraction"],
    )
    rows["objective_min_so_far"] = rows.groupby(group_cols, sort=False)[
        "objective_value_by_fraction"
    ].cummin()
    rows["objective_recovery_from_min"] = (
        rows["objective_value_by_fraction"] - rows["objective_min_so_far"]
    )
    return rows, int(specs["local_pair_id"].nunique())


def _adjacent_fraction_band(fraction_orders: list[int]) -> bool:
    if len(fraction_orders) < 2:
        return False
    ordered = sorted(fraction_orders)
    return max(ordered) - min(ordered) == len(ordered) - 1


def _mechanism_sequence(ordered: pd.DataFrame) -> str:
    return " -> ".join(
        (
            f"{float(row.bridge_edge_weight_fraction):.5g}:"
            f"{str(row.mechanism_read)}"
        )
        for row in ordered.itertuples(index=False)
    )


def _seed_route_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_sequence_id",
        "local_pair_id",
        "next_contract_role",
        "start_condition",
        "start_condition_macro_role",
        "seed",
    ]
    for keys, group in trace_rows.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        ordered = group.sort_values("fraction_order", kind="mergesort")
        first = ordered.iloc[0]
        final = ordered.iloc[-1]
        source_rows = ordered[
            ordered["mechanism_read"].astype(str).isin(SOURCE_FAMILY_MECHANISMS)
        ]
        single_side_rows = ordered[
            ordered["mechanism_read"].astype(str).eq(SINGLE_SIDE_MECHANISM)
        ]
        target_rows = ordered[
            ordered["mechanism_read"].astype(str).eq(TARGET_LIKE_MECHANISM)
        ]
        single_side_orders = [
            int(value) for value in single_side_rows["fraction_order"].tolist()
        ]
        source_family_start = str(first["mechanism_read"]) in SOURCE_FAMILY_MECHANISMS
        final_target_like = str(final["mechanism_read"]) == TARGET_LIKE_MECHANISM
        finite_single_side_band = _adjacent_fraction_band(single_side_orders)
        diagnostic_recurrence = (
            source_family_start
            and finite_single_side_band
            and final_target_like
        )
        if diagnostic_recurrence:
            route_class = "diagnostic_transition_band_recurrence"
        elif not source_family_start:
            route_class = "source_family_start_absent"
        elif not single_side_orders and final_target_like:
            route_class = "source_to_target_without_single_side_band"
        elif single_side_orders and not finite_single_side_band:
            route_class = "single_side_nonfinite_or_fragmented"
        elif single_side_orders and not final_target_like:
            route_class = "single_side_band_without_target_final"
        elif not final_target_like:
            route_class = "target_like_final_absent"
        else:
            route_class = "other_gap_fill_route_failure"
        objective_values = list(map(float, ordered["objective_value_by_fraction"].tolist()))
        objective_diffs = [
            objective_values[index + 1] - objective_values[index]
            for index in range(len(objective_values) - 1)
        ]
        rows.append(
            {
                **key_data,
                "fraction_count": int(len(ordered)),
                "source_family_start": bool(source_family_start),
                "source_family_fraction_count": int(len(source_rows)),
                "source_family_fractions": ";".join(
                    f"{float(value):.5g}"
                    for value in source_rows["bridge_edge_weight_fraction"].tolist()
                ),
                "single_side_fraction_count": int(len(single_side_rows)),
                "single_side_fractions": ";".join(
                    f"{float(value):.5g}"
                    for value in single_side_rows["bridge_edge_weight_fraction"].tolist()
                ),
                "single_side_adjacent_fraction_band": bool(finite_single_side_band),
                "target_like_fraction_count": int(len(target_rows)),
                "target_like_fractions": ";".join(
                    f"{float(value):.5g}"
                    for value in target_rows["bridge_edge_weight_fraction"].tolist()
                ),
                "final_target_like": bool(final_target_like),
                "distinct_mechanism_read_count": int(ordered["mechanism_read"].nunique()),
                "mechanism_read_sequence": _mechanism_sequence(ordered),
                "objective_monotone_nonincreasing_with_bridge_release": bool(
                    all(delta <= EPS for delta in objective_diffs)
                ),
                "max_objective_debt_from_start": float(
                    ordered["objective_debt_from_start"].max()
                ),
                "max_objective_recovery_from_min": float(
                    ordered["objective_recovery_from_min"].max()
                ),
                "diagnostic_recurrence_pass": bool(diagnostic_recurrence),
                "gap_fill_route_class": route_class,
                "wall_claim_ready_after_trace": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _pair_readout_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id, group in seed_rows.groupby("local_pair_id", sort=False):
        route_count = int(len(group))
        pass_count = int(group["diagnostic_recurrence_pass"].map(_as_bool).sum())
        source_count = int(group["source_family_start"].map(_as_bool).sum())
        band_count = int(group["single_side_adjacent_fraction_band"].map(_as_bool).sum())
        target_count = int(group["final_target_like"].map(_as_bool).sum())
        if pass_count == route_count and route_count > 0:
            pair_class = "candidate_full_diagnostic_recurrence"
        elif pass_count > 0:
            pair_class = "candidate_partial_diagnostic_recurrence"
        elif route_count > 0:
            pair_class = "candidate_scoreable_negative_no_recurrence"
        else:
            pair_class = "candidate_residual_gap_no_route_readout"
        rows.append(
            {
                "local_pair_id": str(pair_id),
                "route_sequence_count": route_count,
                "diagnostic_recurrence_pass_count": pass_count,
                "source_family_start_count": source_count,
                "finite_single_side_band_count": band_count,
                "final_target_like_count": target_count,
                "gap_fill_route_class_counts": group[
                    "gap_fill_route_class"
                ].value_counts().to_dict(),
                "gap_fill_pair_class": pair_class,
                "wall_claim_ready_after_trace": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _fraction_readout_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in trace_rows.groupby(
        ["local_pair_id", "start_condition", "bridge_edge_weight_fraction"],
        sort=False,
    ):
        pair_id, start_condition, fraction = keys
        mechanism_counts = group["mechanism_read"].astype(str).value_counts()
        dominant = str(mechanism_counts.index[0]) if not mechanism_counts.empty else ""
        rows.append(
            {
                "local_pair_id": str(pair_id),
                "start_condition": str(start_condition),
                "bridge_edge_weight_fraction": float(fraction),
                "trace_row_count": int(len(group)),
                "source_family_count": int(
                    group["mechanism_read"].astype(str).isin(SOURCE_FAMILY_MECHANISMS).sum()
                ),
                "single_side_count": int(
                    group["mechanism_read"].astype(str).eq(SINGLE_SIDE_MECHANISM).sum()
                ),
                "target_like_count": int(
                    group["mechanism_read"].astype(str).eq(TARGET_LIKE_MECHANISM).sum()
                ),
                "dominant_mechanism_read": dominant,
                "mechanism_read_counts": mechanism_counts.to_dict(),
                "objective_value_mean": float(group["objective_value_by_fraction"].mean()),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["local_pair_id", "start_condition", "bridge_edge_weight_fraction"],
        ascending=[True, True, False],
        kind="mergesort",
    )


def _gate_matrix(
    *,
    contract_gates: pd.DataFrame,
    route_plan: pd.DataFrame,
    trace_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    seeds: int,
) -> pd.DataFrame:
    expected_trace_rows = int(len(route_plan)) * int(seeds)
    route_sequence_count = int(
        route_plan[["local_pair_id", "start_condition"]].drop_duplicates().shape[0]
    ) * int(seeds)
    return pd.DataFrame(
        [
            _gate_row(
                "G1_contract_gates_pass",
                "Did every upstream gap-fill contract gate pass?",
                _count_dict(contract_gates["gate_status"]),
                "all contract gates pass",
                bool(contract_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_exact_fraction_row_scope",
                "Was execution restricted to the 54 predeclared fraction rows?",
                {
                    "route_plan_rows": int(len(route_plan)),
                    "executed_route_contracts": int(trace_rows["route_contract_id"].nunique()),
                    "executed_pairs": sorted(trace_rows["local_pair_id"].astype(str).unique()),
                },
                "54 fraction-expanded route rows, only 001/007",
                int(len(route_plan)) == EXPECTED_ROUTE_PLAN_ROWS
                and int(trace_rows["route_contract_id"].nunique()) == EXPECTED_ROUTE_PLAN_ROWS
                and set(trace_rows["local_pair_id"].astype(str)) == set(GAP_FILL_CANDIDATE_IDS),
            ),
            _gate_row(
                "G3_seed_replicates_complete",
                "Was every route row executed for the requested seed count?",
                {
                    "trace_rows": int(len(trace_rows)),
                    "expected_trace_rows": expected_trace_rows,
                    "seed_count": int(seeds),
                },
                "route_plan_rows * seeds trace rows",
                int(len(trace_rows)) == expected_trace_rows,
            ),
            _gate_row(
                "G4_sequence_readouts_materialized",
                "Were pair/start/seed fraction sequences materialized for classification?",
                {
                    "seed_route_rows": int(len(seed_rows)),
                    "expected_seed_route_rows": route_sequence_count,
                    "route_class_counts": _count_dict(seed_rows["gap_fill_route_class"]),
                },
                "six pair/start routes times seed count",
                int(len(seed_rows)) == route_sequence_count,
            ),
            _gate_row(
                "G5_pair_classification_materialized",
                "Were 001 and 007 classified at pair level?",
                pair_rows[["local_pair_id", "gap_fill_pair_class"]].to_dict("records"),
                "two pair-level readout rows",
                set(pair_rows["local_pair_id"].astype(str)) == set(GAP_FILL_CANDIDATE_IDS)
                and len(pair_rows) == len(GAP_FILL_CANDIDATE_IDS),
            ),
            _gate_row(
                "G6_no_claim_promotion",
                "Are wall, pathway, method, quality, replay, and generality claims closed?",
                {
                    "wall_claim_ready_after_trace": _count_dict(
                        seed_rows["wall_claim_ready_after_trace"]
                    ),
                    "wall_promotion_status": _count_dict(seed_rows["wall_promotion_status"]),
                    "method_status": _count_dict(seed_rows["method_status"]),
                },
                "all claim flags remain false/diagnostic-only",
                bool(seed_rows["wall_claim_ready_after_trace"].eq(False).all())
                and bool(seed_rows["wall_promotion_status"].astype(str).eq(WALL_PROMOTION_STATUS).all())
                and bool(seed_rows["method_status"].astype(str).eq(METHOD_STATUS).all()),
            ),
        ]
    )


def _summary(
    *,
    contract_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    route_plan: pd.DataFrame,
    trace_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    gates: pd.DataFrame,
    seeds: int,
) -> dict[str, Any]:
    pair_classes = _count_dict(pair_rows["gap_fill_pair_class"])
    any_recurrence = bool(pair_rows["diagnostic_recurrence_pass_count"].astype(int).gt(0).any())
    if any_recurrence:
        recommended_next_gate = (
            "Audit 001/007 positive recurrence rows under the surface-claim schema; "
            "keep wall/pathway/panel-generality claims closed until object evidence "
            "and independent recurrence controls are predeclared."
        )
    else:
        recommended_next_gate = (
            "Audit 001/007 as scoreable negatives or residual gaps, then decide "
            "whether the 17-gap panel still needs more route/fraction readout or "
            "whether the six-row core should remain the fixed guard surface."
        )
    return {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_summary.v1",
        "status": RUN_STATUS,
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "route_plan_row_count": int(len(route_plan)),
        "trace_row_count": int(len(trace_rows)),
        "seed_route_row_count": int(len(seed_rows)),
        "pair_readout_row_count": int(len(pair_rows)),
        "fraction_readout_row_count": int(len(fraction_rows)),
        "seed_count": int(seeds),
        "candidate_pair_ids": sorted(pair_rows["local_pair_id"].astype(str).tolist()),
        "gap_fill_pair_class_counts": pair_classes,
        "diagnostic_recurrence_pair_ids": sorted(
            pair_rows.loc[
                pair_rows["diagnostic_recurrence_pass_count"].astype(int).gt(0),
                "local_pair_id",
            ].astype(str).tolist()
        ),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].astype(str).tolist(),
        "route_execution_opened": True,
        "panel_generality_claim_ready": False,
        "wall_claim_ready": False,
        "pathway_claim_ready": False,
        "method_claim_ready": False,
        "quality_claim_ready": False,
        "interpretation": (
            "The predeclared 001/007 gap-fill trace has executed locally. The result "
            "is a route/fraction readout only; it can support diagnostic recurrence, "
            "scoreable-negative, or residual-gap wording after audit, but not wall, "
            "pathway, method, quality/cost, full-replay, or panel-generality claims."
        ),
        "recommended_next_gate": recommended_next_gate,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Surface Rule Gap-Fill Trace",
        "",
        f"- status: `{summary['status']}`",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- seed_route_row_count: {summary['seed_route_row_count']}",
        f"- candidate_pair_ids: {summary['candidate_pair_ids']}",
        f"- gap_fill_pair_class_counts: {summary['gap_fill_pair_class_counts']}",
        f"- diagnostic_recurrence_pair_ids: {summary['diagnostic_recurrence_pair_ids']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Readout Rows",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "route_sequence_count",
                "diagnostic_recurrence_pass_count",
                "source_family_start_count",
                "finite_single_side_band_count",
                "final_target_like_count",
                "gap_fill_route_class_counts",
                "gap_fill_pair_class",
            ],
        ),
        "",
        "## Seed Route Rows",
        "",
        _markdown_table(
            seed_rows,
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "source_family_start",
                "single_side_fraction_count",
                "final_target_like",
                "diagnostic_recurrence_pass",
                "gap_fill_route_class",
                "mechanism_read_sequence",
            ],
        ),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gates,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
        ),
        "",
        "## Boundary",
        "",
        (
            "This is a local gap-fill trace. It materializes 001/007 readout rows, "
            "not wall/pathway, panel-generality, quality/cost, full-replay, or "
            "method evidence."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract_dir = Path(args.contract_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_plan = _read_csv(contract_dir / CONTRACT_ROUTE_PLAN_ROWS_CSV)
    contract_gates = _read_csv(contract_dir / CONTRACT_GATE_MATRIX_CSV)
    trace_rows, candidate_pair_count = _trace_rows(
        route_plan=route_plan,
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        gamma=float(args.gamma),
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
        edge_chunk_size=int(args.edge_chunk_size),
    )
    seed_rows = _seed_route_rows(trace_rows)
    pair_rows = _pair_readout_rows(seed_rows)
    fraction_rows = _fraction_readout_rows(trace_rows)
    gates = _gate_matrix(
        contract_gates=contract_gates,
        route_plan=route_plan,
        trace_rows=trace_rows,
        seed_rows=seed_rows,
        pair_rows=pair_rows,
        seeds=int(args.seeds),
    )
    summary = _summary(
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        route_plan=route_plan,
        trace_rows=trace_rows,
        seed_rows=seed_rows,
        pair_rows=pair_rows,
        fraction_rows=fraction_rows,
        gates=gates,
        seeds=int(args.seeds),
    )
    summary["candidate_pair_count"] = int(candidate_pair_count)

    _write_csv(trace_rows, output_dir / TRACE_ROWS_CSV)
    _write_csv(seed_rows, output_dir / SEED_ROUTE_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_READOUT_ROWS_CSV)
    _write_csv(fraction_rows, output_dir / FRACTION_READOUT_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_config.v1",
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "edge_chunk_size": int(args.edge_chunk_size),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_rows=pair_rows,
        seed_rows=seed_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gamma", type=float, default=1.0e-5)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--edge-chunk-size", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
