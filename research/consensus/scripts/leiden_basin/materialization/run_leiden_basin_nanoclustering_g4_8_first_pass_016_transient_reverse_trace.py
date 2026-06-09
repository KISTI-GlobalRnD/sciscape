#!/usr/bin/env python3
"""Execute the local_pair_016 same-seed target-anchor reverse trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace as scoped_trace
from design_leiden_basin_nanoclustering_g4_8_first_pass_016_transient_reverse_contract import (
    CLAIM_BOUNDARY as CONTRACT_CLAIM_BOUNDARY,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTRACT_DIR,
    FRACTION_STEP_ROWS_CSV as CONTRACT_FRACTION_STEP_ROWS_CSV,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    PLANNED_ROUTE_FAMILY,
    PRIMARY_PAIR_ID,
    ROUTE_PLAN_ROWS_CSV as CONTRACT_ROUTE_PLAN_ROWS_CSV,
    TARGET_SIGNATURE_ID,
    TRANSIENT_SIGNATURE_ID,
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
    SEED_RUNS_CSV as LOCAL_ABLATION_SEED_RUNS_CSV,
    _build_igraph,
    _canonical_groups,
    _collect_induced_edges_by_branch,
    _mechanism_read,
    _node_doc_lookup,
    _read_json,
    _signature_id,
)
from sciscape.clustering.runner import LeidenRunner


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_reverse_trace_gamma1e5_20260605"
)

TRACE_ROWS_CSV = "nanoclustering_g4_8_first_pass_016_transient_reverse_trace_rows.csv"
ROUTE_REVERSIBILITY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_reverse_route_rows.csv"
)
FRACTION_SUMMARY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_reverse_fraction_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_016_transient_reverse_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_transient_reverse_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_transient_reverse_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_transient_reverse_report.md"

RUN_STATUS = "executed_nanoclustering_g4_8_first_pass_016_transient_reverse_trace"
ROUTE_EXECUTION_STATUS = "executed_016_same_seed_target_anchor_reverse_trace"
WALL_PROMOTION_STATUS = "not_promoted_016_reverse_trace_only"
METHOD_STATUS = "reverse_trace_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 same-seed target-anchor "
    "reverse trace only; executes the predeclared ascending bridge-fraction "
    "scan from the drop-bridge target anchor. It does not promote basin walls, "
    "replay full NanoClustering, evaluate quality/cost value, or claim "
    "method/algorithm success."
)

EPS = 1e-9
SUPPORT_ANCHOR_DISTANCE_COLUMNS = (
    "support_distance_to_original",
    "support_distance_to_drop_bridge_edges",
    "support_distance_to_drop_direct_edge",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(int(max_rows))
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


def _support_equidistant_to_three_anchors(row: pd.Series) -> bool:
    values: list[float] = []
    for column in SUPPORT_ANCHOR_DISTANCE_COLUMNS:
        value = row.get(column)
        if value is None or pd.isna(value):
            return False
        values.append(float(value))
    return max(values) - min(values) <= EPS


def _endpoint_class(row: pd.Series) -> str:
    signature_id = str(row["result_endpoint_signature_id"])
    assignment = str(row["endpoint_assignment_by_step"])
    if signature_id == TARGET_SIGNATURE_ID or "drop_bridge_target_anchor" in assignment:
        return "target_anchor"
    if signature_id == TRANSIENT_SIGNATURE_ID:
        return "transient_signature"
    if _as_bool(row.get("matches_original_anchor", False)):
        return "source_anchor"
    if assignment == "unknown_new_endpoint":
        return "unknown_other"
    return "other_anchor"


def _target_anchor_trace_rows(
    *,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    contract_dir: Path,
    local_ablation_dir: Path,
    gamma: float,
    seeds: int,
    n_iterations: int,
    edge_chunk_size: int,
) -> tuple[pd.DataFrame, int, int]:
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
            scoped_trace._parse_node_ids(spec.local_node_ids)
        )
    induced_edges_by_branch = _collect_induced_edges_by_branch(
        graph_mechanism_dir=graph_mechanism_dir,
        target_nodes_by_branch=target_nodes_by_branch,
        edge_chunk_size=int(edge_chunk_size),
    )

    seed_runs = _read_csv(local_ablation_dir / LOCAL_ABLATION_SEED_RUNS_CSV)
    anchors_by_key = scoped_trace._anchor_lookup(seed_runs, route_plan)
    spec_by_pair = {
        str(row.local_pair_id): row._asdict()
        for row in specs.sort_values("local_pair_id", kind="mergesort").itertuples(index=False)
    }
    output_rows: list[dict[str, Any]] = []

    for step_row in fraction_steps.sort_values(
        ["route_contract_id", "step_index"],
        kind="mergesort",
    ).itertuples(index=False):
        step = step_row._asdict()
        local_pair_id = str(step["local_pair_id"])
        spec = spec_by_pair[local_pair_id]
        object_role_id = str(spec["object_role_universe_id"])
        branch = str(spec["branch"])
        left = int(spec["left_node_id"])
        right = int(spec["right_node_id"])
        node_ids = scoped_trace._parse_node_ids(spec["local_node_ids"])
        bridge_nodes = set(scoped_trace._parse_node_ids(spec["selected_bridge_node_ids"]))
        node_sizes = [
            int(doc_lookup.get((object_role_id, int(node_id)), 1))
            for node_id in node_ids
        ]
        induced_edges = induced_edges_by_branch.get(
            branch,
            pd.DataFrame(columns=["source", "target", "weight"]),
        )
        local_edges = scoped_trace._scaled_local_edges(
            induced_edges=induced_edges,
            node_ids=node_ids,
            left_node=left,
            right_node=right,
            bridge_nodes=bridge_nodes,
            direct_fraction=float(step["direct_edge_weight_fraction"]),
            bridge_fraction=float(step["bridge_edge_weight_fraction"]),
        )
        edge_parts = scoped_trace._edge_weight_parts(
            induced_edges=induced_edges,
            node_ids=node_ids,
            left_node=left,
            right_node=right,
            bridge_nodes=bridge_nodes,
        )
        graph = _build_igraph(node_ids, local_edges)
        runner = LeidenRunner(graph, objective="cpm", default_iterations=int(n_iterations))
        for seed in range(int(seeds)):
            anchors: dict[str, dict[str, Any]] = {}
            for variant in scoped_trace.GRAPH_VARIANT_ORDER:
                anchor = anchors_by_key.get(
                    (local_pair_id, str(step["start_condition"]), int(seed), variant)
                )
                if anchor is None:
                    continue
                anchors[variant] = {
                    **anchor,
                    "membership": scoped_trace._groups_to_membership(
                        node_ids,
                        str(anchor["endpoint_signature"]),
                    ),
                }
            initial_anchor = anchors.get(str(step["initial_anchor_variant"]))
            if initial_anchor is None:
                raise ValueError(
                    "missing reverse initial anchor "
                    f"{step['initial_anchor_variant']} for {local_pair_id} "
                    f"{step['start_condition']} seed={seed}"
                )
            initial_membership = list(map(int, initial_anchor["membership"]))
            initial_signature = _signature_id(_canonical_groups(node_ids, initial_membership))
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
            anchor_data = scoped_trace._anchor_match_data(
                anchors=anchors,
                result_signature_id=result_signature_id,
                result_membership=membership,
                expected_final_variant=str(step["expected_final_anchor_variant"]),
            )
            output_rows.append(
                {
                    "route_trace_row_id": (
                        f"{step['route_contract_id']}__seed{seed:02d}"
                        f"__step{int(step['step_index']):02d}"
                    ),
                    "route_contract_id": str(step["route_contract_id"]),
                    "validation_unit_id": str(step["validation_unit_id"]),
                    "local_pair_id": local_pair_id,
                    "branch": branch,
                    "left_node_id": left,
                    "right_node_id": right,
                    "object_role_universe_id": object_role_id,
                    "start_condition": str(step["start_condition"]),
                    "planned_route_family": str(step["planned_route_family"]),
                    "route_family_role": str(step["route_family_role"]),
                    "step_index": int(step["step_index"]),
                    "step_label": str(step["step_label"]),
                    "seed": int(seed),
                    "gamma": float(gamma),
                    "n_iterations": int(n_iterations),
                    "direct_edge_weight_fraction": float(step["direct_edge_weight_fraction"]),
                    "bridge_edge_weight_fraction": float(step["bridge_edge_weight_fraction"]),
                    "initial_anchor_variant": str(step["initial_anchor_variant"]),
                    "expected_final_anchor_variant": str(step["expected_final_anchor_variant"]),
                    "local_node_count": int(len(node_ids)),
                    "selected_bridge_count": int(len(bridge_nodes)),
                    "local_edge_count": int(graph.ecount()),
                    "local_edge_weight_sum": float(sum(graph.es["weight"])) if graph.ecount() else 0.0,
                    "active_direct_edge_weight": float(
                        edge_parts["original_direct_edge_weight"]
                        * float(step["direct_edge_weight_fraction"])
                    ),
                    "active_pair_bridge_edge_weight_sum": float(
                        edge_parts["original_pair_bridge_edge_weight_sum"]
                        * float(step["bridge_edge_weight_fraction"])
                    ),
                    **edge_parts,
                    "initial_endpoint_signature_id": initial_signature,
                    "initial_anchor_signature_id": str(initial_anchor["endpoint_signature_id"]),
                    "initial_anchor_assignment": scoped_trace.ANCHOR_VARIANT_TO_ASSIGNMENT[
                        str(step["initial_anchor_variant"])
                    ],
                    "initial_matches_requested_anchor": bool(
                        initial_signature == str(initial_anchor["endpoint_signature_id"])
                    ),
                    "result_endpoint_signature_id": result_signature_id,
                    "result_endpoint_signature": json.dumps(groups, sort_keys=True),
                    "objective_value_by_step": float(result.quality),
                    "cluster_count": int(result.cluster_count),
                    **read,
                    **anchor_data,
                    "support_distance_by_step": anchor_data["support_distance_min_known_anchor"],
                    "polish_changed_from_initial": result_signature_id != initial_signature,
                    "polish_reverted_to_original_anchor": bool(
                        anchor_data["matches_original_anchor"]
                    ),
                    "polish_reversion_check": bool(anchor_data["matches_original_anchor"]),
                    "post_route_endpoint_assignment_available": True,
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "run_status": RUN_STATUS,
                    "wall_generality_claim_allowed_after_trace": False,
                    "method_claim_allowed_after_trace": False,
                    "quality_cost_claim_allowed_after_trace": False,
                    "contract_dir": str(contract_dir),
                    "local_ablation_dir": str(local_ablation_dir),
                }
            )
    rows = pd.DataFrame(output_rows)
    if rows.empty:
        return rows, int(len(fraction_steps)), int(specs["local_pair_id"].nunique())
    rows = rows.sort_values(
        ["route_contract_id", "seed", "step_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    group_cols = ["route_contract_id", "seed"]
    rows["objective_start_value"] = rows.groupby(group_cols, sort=False)[
        "objective_value_by_step"
    ].transform("first")
    rows["objective_delta_from_start"] = (
        rows["objective_value_by_step"] - rows["objective_start_value"]
    )
    rows["objective_debt_from_start"] = (rows["objective_start_value"] - rows["objective_value_by_step"]).clip(lower=0.0)
    rows["objective_min_so_far"] = rows.groupby(group_cols, sort=False)[
        "objective_value_by_step"
    ].cummin()
    rows["objective_recovery_from_min"] = (
        rows["objective_value_by_step"] - rows["objective_min_so_far"]
    )
    return rows, int(len(fraction_steps)), int(specs["local_pair_id"].nunique())


def _route_reversibility_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trace_rows.empty:
        return pd.DataFrame(rows)
    classified = trace_rows.copy()
    classified["endpoint_class"] = classified.apply(_endpoint_class, axis=1)
    group_cols = [
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "planned_route_family",
        "seed",
    ]
    for keys, group in classified.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        ordered = group.sort_values("step_index", kind="mergesort").copy()
        target = ordered[ordered["endpoint_class"].eq("target_anchor")]
        transient = ordered[ordered["endpoint_class"].eq("transient_signature")]
        source = ordered[ordered["endpoint_class"].eq("source_anchor")]
        first = ordered.iloc[0]
        final = ordered.iloc[-1]
        first_is_target = str(first["endpoint_class"]) == "target_anchor"
        final_is_source = str(final["endpoint_class"]) == "source_anchor"
        final_is_target = str(final["endpoint_class"]) == "target_anchor"
        if first_is_target and final_is_source:
            reverse_class = "target_to_source_reversal_observed"
        elif first_is_target and final_is_target:
            reverse_class = "target_hysteresis_persists_to_full_bridge"
        elif first_is_target and len(transient) > 0 and not final_is_source:
            reverse_class = "target_to_transient_without_source_return"
        elif not first_is_target:
            reverse_class = "target_anchor_initialization_failed"
        else:
            reverse_class = "mixed_reverse_endpoint_sequence"
        endpoint_sequence = " -> ".join(
            f"{float(row.bridge_edge_weight_fraction):.5g}:{row.endpoint_class}"
            for row in ordered.itertuples(index=False)
        )
        objective_values = list(map(float, ordered["objective_value_by_step"].tolist()))
        objective_diffs = [
            objective_values[index + 1] - objective_values[index]
            for index in range(len(objective_values) - 1)
        ]
        rows.append(
            {
                **key_data,
                "route_key": f"{key_data['start_condition']}|seed={int(key_data['seed'])}",
                "fraction_count": int(len(ordered)),
                "target_fraction_count": int(len(target)),
                "target_fractions": ";".join(
                    f"{float(value):.5g}" for value in target["bridge_edge_weight_fraction"].tolist()
                ),
                "transient_fraction_count": int(len(transient)),
                "transient_fractions": ";".join(
                    f"{float(value):.5g}" for value in transient["bridge_edge_weight_fraction"].tolist()
                ),
                "source_fraction_count": int(len(source)),
                "source_fractions": ";".join(
                    f"{float(value):.5g}" for value in source["bridge_edge_weight_fraction"].tolist()
                ),
                "first_endpoint_class": str(first["endpoint_class"]),
                "final_endpoint_class": str(final["endpoint_class"]),
                "endpoint_class_sequence": endpoint_sequence,
                "distinct_signature_count": int(ordered["result_endpoint_signature_id"].nunique()),
                "initial_matches_requested_anchor_all_steps": bool(
                    ordered["initial_matches_requested_anchor"].astype(bool).all()
                ),
                "objective_monotone_nonincreasing_with_bridge_restore": bool(
                    all(delta <= EPS for delta in objective_diffs)
                ),
                "max_objective_debt_from_start": float(ordered["objective_debt_from_start"].max()),
                "max_objective_recovery_from_min": float(
                    ordered["objective_recovery_from_min"].max()
                ),
                "reverse_trace_class": reverse_class,
                "wall_claim_ready_after_trace": False,
                "wall_claim_block_reason": (
                    "reverse trace materializes target-initialized path shape only; "
                    "wall promotion still needs explicit basin relation and "
                    "objective/barrier interpretation"
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _fraction_summary_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    if trace_rows.empty:
        return pd.DataFrame()
    classified = trace_rows.copy()
    classified["endpoint_class"] = classified.apply(_endpoint_class, axis=1)
    classified["support_equidistant_to_three_primary_anchors"] = classified.apply(
        _support_equidistant_to_three_anchors,
        axis=1,
    )
    rows: list[dict[str, Any]] = []
    for fraction, group in classified.groupby("bridge_edge_weight_fraction", sort=False):
        signature_counts = group["result_endpoint_signature_id"].astype(str).value_counts()
        dominant_signature_id = str(signature_counts.index[0]) if not signature_counts.empty else ""
        dominant_count = int(signature_counts.iloc[0]) if not signature_counts.empty else 0
        rows.append(
            {
                "bridge_edge_weight_fraction": float(fraction),
                "trace_row_count": int(len(group)),
                "route_count": int(
                    group[["route_contract_id", "seed"]].drop_duplicates().shape[0]
                ),
                "target_anchor_count": int(group["endpoint_class"].eq("target_anchor").sum()),
                "transient_signature_count": int(
                    group["endpoint_class"].eq("transient_signature").sum()
                ),
                "source_anchor_count": int(group["endpoint_class"].eq("source_anchor").sum()),
                "unknown_other_count": int(group["endpoint_class"].eq("unknown_other").sum()),
                "other_anchor_count": int(group["endpoint_class"].eq("other_anchor").sum()),
                "distinct_signature_count": int(group["result_endpoint_signature_id"].nunique()),
                "dominant_signature_id": dominant_signature_id,
                "dominant_signature_count": dominant_count,
                "support_equidistant_to_three_primary_anchors_count": int(
                    group["support_equidistant_to_three_primary_anchors"].astype(bool).sum()
                ),
                "objective_value_mean": float(group["objective_value_by_step"].mean()),
                "objective_value_min": float(group["objective_value_by_step"].min()),
                "objective_value_max": float(group["objective_value_by_step"].max()),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "bridge_edge_weight_fraction",
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)


def _reverse_semantic_class(route_rows: pd.DataFrame) -> str:
    if route_rows.empty:
        return "not_classified_no_route_rows"
    expected = int(len(route_rows))
    counts = route_rows["reverse_trace_class"].value_counts().to_dict()
    if counts.get("target_to_source_reversal_observed", 0) == expected:
        return "fully_reversible_target_to_source_under_same_seed_anchor"
    if counts.get("target_hysteresis_persists_to_full_bridge", 0) == expected:
        return "full_target_hysteresis_under_same_seed_anchor"
    if counts.get("target_to_transient_without_source_return", 0) == expected:
        return "reverse_reaches_transient_but_not_source"
    return "mixed_reverse_reversibility_class"


def _gate_matrix(
    *,
    contract_gates: pd.DataFrame,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    trace_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    step_config_count: int,
    seeds: int,
) -> pd.DataFrame:
    expected_trace_rows = int(step_config_count) * int(seeds)
    expected_seed_routes = int(len(route_plan)) * int(seeds)
    start_rows = trace_rows[trace_rows["step_index"].astype(int).eq(1)]
    final_rows = trace_rows[trace_rows["step_index"].astype(int).eq(9)]
    target_start_count = int(start_rows.apply(_endpoint_class, axis=1).eq("target_anchor").sum())
    classified_count = int(len(route_rows))
    return pd.DataFrame(
        [
            _gate_row(
                "G1_contract_gates_pass",
                "Did every upstream 016 reverse contract gate pass?",
                contract_gates["gate_status"].value_counts().to_dict(),
                "all contract gates pass",
                bool(contract_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_exact_reverse_trace_scope",
                "Was reverse execution restricted to the predeclared route/fraction/seed grid?",
                {
                    "route_plan_rows": len(route_plan),
                    "step_config_count": step_config_count,
                    "seeds": seeds,
                    "trace_rows": len(trace_rows),
                    "expected_trace_rows": expected_trace_rows,
                    "executed_pairs": sorted(trace_rows["local_pair_id"].astype(str).unique().tolist()),
                },
                "3 route rows * 9 fractions * 8 seeds = 216 trace rows, only local_pair_016",
                len(route_plan) == 3
                and int(step_config_count) == 27
                and len(trace_rows) == expected_trace_rows
                and set(trace_rows["local_pair_id"].astype(str)) == {PRIMARY_PAIR_ID},
            ),
            _gate_row(
                "G3_target_anchor_initialization_materialized",
                "Does the first reverse step reconcile to the target anchor in every route?",
                {
                    "start_rows": len(start_rows),
                    "target_start_count": target_start_count,
                    "initial_anchor_matches": int(
                        start_rows["initial_matches_requested_anchor"].astype(bool).sum()
                    ),
                },
                "all 24 start rows initialize and remain at drop-bridge target anchor",
                len(start_rows) == expected_seed_routes
                and target_start_count == expected_seed_routes
                and bool(start_rows["initial_matches_requested_anchor"].astype(bool).all()),
            ),
            _gate_row(
                "G4_reverse_sequence_classified",
                "Was every seed route assigned a reverse sequence class?",
                {
                    "classified_count": classified_count,
                    "reverse_trace_class_counts": route_rows["reverse_trace_class"].value_counts().to_dict()
                    if not route_rows.empty
                    else {},
                    "fraction_rows": fraction_rows[
                        [
                            "bridge_edge_weight_fraction",
                            "target_anchor_count",
                            "transient_signature_count",
                            "source_anchor_count",
                            "dominant_signature_id",
                        ]
                    ].to_dict("records")
                    if not fraction_rows.empty
                    else [],
                },
                "all 24 reverse seed routes classified",
                classified_count == expected_seed_routes,
            ),
            _gate_row(
                "G5_final_source_return_observed",
                "Does the target-initialized reverse trace return to the source anchor at full bridge weight?",
                {
                    "final_rows": len(final_rows),
                    "final_source_count": int(final_rows.apply(_endpoint_class, axis=1).eq("source_anchor").sum()),
                    "final_target_count": int(final_rows.apply(_endpoint_class, axis=1).eq("target_anchor").sum()),
                },
                "all 24 final rows match source anchor",
                len(final_rows) == expected_seed_routes
                and int(final_rows.apply(_endpoint_class, axis=1).eq("source_anchor").sum())
                == expected_seed_routes,
            ),
            _gate_row(
                "G6_claim_boundaries_closed",
                "Are wall, method, quality/cost, and full replay claims closed?",
                {
                    "wall_flags_all_false": bool(
                        trace_rows["wall_generality_claim_allowed_after_trace"].eq(False).all()
                    ),
                    "method_flags_all_false": bool(
                        trace_rows["method_claim_allowed_after_trace"].eq(False).all()
                    ),
                    "quality_flags_all_false": bool(
                        trace_rows["quality_cost_claim_allowed_after_trace"].eq(False).all()
                    ),
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "contract_boundary": CONTRACT_CLAIM_BOUNDARY,
                },
                "all claim flags false and wall promotion status closed",
                bool(trace_rows["wall_generality_claim_allowed_after_trace"].eq(False).all())
                and bool(trace_rows["method_claim_allowed_after_trace"].eq(False).all())
                and bool(trace_rows["quality_cost_claim_allowed_after_trace"].eq(False).all()),
            ),
        ]
    )


def _summary(
    *,
    contract_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    trace_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    gates: pd.DataFrame,
    step_config_count: int,
    candidate_pair_count: int,
    seeds: int,
) -> dict[str, Any]:
    reverse_class = _reverse_semantic_class(route_rows)
    if reverse_class == "fully_reversible_target_to_source_under_same_seed_anchor":
        next_gate = (
            "Compare forward and reverse transition thresholds to define a "
            "candidate pathway relation; still keep wall language closed until "
            "objective/barrier interpretation is explicit."
        )
    elif reverse_class == "full_target_hysteresis_under_same_seed_anchor":
        next_gate = (
            "Treat this as strong path-asymmetry/hysteresis evidence and design "
            "a threshold-localization audit; do not promote wall language yet."
        )
    else:
        next_gate = (
            "Inspect mixed reverse classes by start and seed before threshold "
            "localization or broader controls."
        )
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_transient_reverse_summary.v1",
        "status": RUN_STATUS,
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "planned_route_family": PLANNED_ROUTE_FAMILY,
        "candidate_pair_count": int(candidate_pair_count),
        "route_plan_row_count": int(len(route_plan)),
        "fraction_step_row_count": int(len(fraction_steps)),
        "route_step_config_count": int(step_config_count),
        "seed_count": int(seeds),
        "trace_row_count": int(len(trace_rows)),
        "route_reversibility_row_count": int(len(route_rows)),
        "fraction_summary_row_count": int(len(fraction_rows)),
        "reverse_semantic_class": reverse_class,
        "reverse_trace_class_counts": route_rows["reverse_trace_class"].value_counts().to_dict()
        if not route_rows.empty
        else {},
        "fraction_class_counts": {
            f"{float(row.bridge_edge_weight_fraction):.5g}": {
                "target": int(row.target_anchor_count),
                "transient": int(row.transient_signature_count),
                "source": int(row.source_anchor_count),
                "unknown": int(row.unknown_other_count),
            }
            for row in fraction_rows.itertuples(index=False)
        }
        if not fraction_rows.empty
        else {},
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "transient_signature_id": TRANSIENT_SIGNATURE_ID,
        "target_signature_id": TARGET_SIGNATURE_ID,
        "interpretation": (
            "The run executes a same-seed target-anchor reverse trace. It "
            "classifies reversibility/path asymmetry only; wall, full replay, "
            "method, and quality/cost claims remain closed."
        ),
        "recommended_next_gate": next_gate,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    path: Path,
    summary: dict[str, Any],
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Transient Reverse Trace",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- reverse_semantic_class: {summary['reverse_semantic_class']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- reverse_trace_class_counts: {summary['reverse_trace_class_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Fraction Summary",
        "",
        _markdown_table(
            fraction_rows,
            [
                "bridge_edge_weight_fraction",
                "route_count",
                "target_anchor_count",
                "transient_signature_count",
                "source_anchor_count",
                "unknown_other_count",
                "dominant_signature_id",
                "objective_value_mean",
            ],
            max_rows=20,
        ),
        "",
        "## Route Reversibility Rows",
        "",
        _markdown_table(
            route_rows,
            [
                "route_key",
                "target_fractions",
                "transient_fractions",
                "source_fractions",
                "first_endpoint_class",
                "final_endpoint_class",
                "reverse_trace_class",
                "max_objective_debt_from_start",
                "max_objective_recovery_from_min",
            ],
            max_rows=40,
        ),
        "",
        "## Gates",
        "",
        _markdown_table(
            gates,
            ["gate_id", "question", "observed", "minimum_or_rule", "gate_status"],
            max_rows=20,
        ),
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Recommended Next Gate",
        "",
        summary["recommended_next_gate"],
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gamma", type=float, default=1e-5)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--edge-chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    contract_dir = Path(args.contract_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_plan = _read_csv(contract_dir / CONTRACT_ROUTE_PLAN_ROWS_CSV)
    fraction_steps = _read_csv(contract_dir / CONTRACT_FRACTION_STEP_ROWS_CSV)
    contract_gates = _read_csv(contract_dir / CONTRACT_GATE_MATRIX_CSV)
    trace_rows, step_config_count, candidate_pair_count = _target_anchor_trace_rows(
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        gamma=float(args.gamma),
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
        edge_chunk_size=int(args.edge_chunk_size),
    )
    route_rows = _route_reversibility_rows(trace_rows)
    fraction_rows = _fraction_summary_rows(trace_rows)
    gates = _gate_matrix(
        contract_gates=contract_gates,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        trace_rows=trace_rows,
        route_rows=route_rows,
        fraction_rows=fraction_rows,
        step_config_count=step_config_count,
        seeds=int(args.seeds),
    )
    summary = _summary(
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        trace_rows=trace_rows,
        route_rows=route_rows,
        fraction_rows=fraction_rows,
        gates=gates,
        step_config_count=step_config_count,
        candidate_pair_count=candidate_pair_count,
        seeds=int(args.seeds),
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_transient_reverse_config.v1",
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "edge_chunk_size": int(args.edge_chunk_size),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(trace_rows, output_dir / TRACE_ROWS_CSV)
    _write_csv(route_rows, output_dir / ROUTE_REVERSIBILITY_ROWS_CSV)
    _write_csv(fraction_rows, output_dir / FRACTION_SUMMARY_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        path=output_dir / REPORT_MD,
        summary=summary,
        route_rows=route_rows,
        fraction_rows=fraction_rows,
        gates=gates,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
