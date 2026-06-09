#!/usr/bin/env python3
"""Audit mechanism interpretation for local_pair_016.

This read-only audit follows the 016 pathway-shape and objective/barrier
audits. It checks whether the source-family transition band has a named local
mechanism under existing local-ablation and trace fields, without rerunning
Leiden or promoting wall/method/quality claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


PRIMARY_PAIR_ID = "local_pair_016"

DEFAULT_LOCAL_ABLATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation_gamma1e5_20260603"
)
DEFAULT_SEMANTIC_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_semantic_validation_gamma1e5_20260605"
)
DEFAULT_PERSISTENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_trace_gamma1e5_20260605"
)
DEFAULT_REVERSE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_reverse_trace_gamma1e5_20260605"
)
DEFAULT_PATHWAY_SHAPE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_pathway_shape_audit_gamma1e5_20260605"
)
DEFAULT_OBJECTIVE_BARRIER_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_objective_barrier_audit_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_mechanism_interpretation_audit_gamma1e5_20260605"
)

LOCAL_SUBSTRATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_mechanism_local_substrate_rows.csv"
)
VARIANT_MECHANISM_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_mechanism_variant_rows.csv"
)
FRACTION_MECHANISM_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_mechanism_fraction_rows.csv"
)
ROUTE_MECHANISM_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_mechanism_route_rows.csv"
)
DECISION_ROWS_CSV = "nanoclustering_g4_8_first_pass_016_mechanism_decision_rows.csv"
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_016_mechanism_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_mechanism_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_mechanism_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_mechanism_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_016_mechanism_interpretation"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_016_mechanism_interpretation"
WALL_PROMOTION_STATUS = "not_promoted_mechanism_interpretation_only"
METHOD_STATUS = "mechanism_interpretation_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 mechanism-interpretation "
    "audit only; reads local-ablation, semantic-validation, pathway-shape, "
    "objective/barrier, and trace artifacts to name a local mechanism. It does "
    "not rerun Leiden, promote basin walls, replay full NanoClustering, "
    "evaluate quality/cost value, or claim method success."
)

TRANSIENT_FRACTIONS = {0.625, 0.6875, 0.71875, 0.75, 0.78125, 0.8125}
SOURCE_FAMILY_FRACTIONS = {0.875, 1.0}
TARGET_FRACTIONS = {0.5}
EPS = 1e-9


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _route_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    if "route_key" not in rows.columns:
        rows["route_key"] = (
            rows["start_condition"].astype(str)
            + "|seed="
            + rows["seed"].astype(int).astype(str)
        )
    return rows


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


def _load_context(
    *,
    local_ablation_dir: Path,
    semantic_dir: Path,
    persistence_dir: Path,
    reverse_dir: Path,
    pathway_shape_dir: Path,
    objective_barrier_dir: Path,
) -> dict[str, Any]:
    return {
        "local_graph": _read_csv(
            local_ablation_dir
            / "nanoclustering_symmetric_object_variable_pair_local_ablation_graph_rows.csv"
        ),
        "variant_summary": _read_csv(
            local_ablation_dir
            / "nanoclustering_symmetric_object_variable_pair_local_ablation_variant_summary.csv"
        ),
        "pair_gates": _read_csv(
            local_ablation_dir
            / "nanoclustering_symmetric_object_variable_pair_local_ablation_pair_gate_rows.csv"
        ),
        "semantic_route": _read_csv(
            semantic_dir
            / "nanoclustering_g4_8_first_pass_016_transient_semantic_route_rows.csv"
        ),
        "semantic_gates": _read_csv(
            semantic_dir
            / "nanoclustering_g4_8_first_pass_016_transient_semantic_gate_matrix.csv"
        ),
        "persistence_trace": _route_key_frame(
            _read_csv(
                persistence_dir
                / "nanoclustering_g4_8_first_pass_016_transient_persistence_trace_rows.csv"
            )
        ),
        "reverse_trace": _route_key_frame(
            _read_csv(
                reverse_dir
                / "nanoclustering_g4_8_first_pass_016_transient_reverse_trace_rows.csv"
            )
        ),
        "pathway_summary": _read_json(
            pathway_shape_dir
            / "nanoclustering_g4_8_first_pass_016_pathway_shape_summary.json"
        ),
        "pathway_gates": _read_csv(
            pathway_shape_dir
            / "nanoclustering_g4_8_first_pass_016_pathway_shape_gate_matrix.csv"
        ),
        "pathway_route": _read_csv(
            pathway_shape_dir
            / "nanoclustering_g4_8_first_pass_016_pathway_shape_route_rows.csv"
        ),
        "objective_summary": _read_json(
            objective_barrier_dir
            / "nanoclustering_g4_8_first_pass_016_objective_barrier_summary.json"
        ),
        "objective_gates": _read_csv(
            objective_barrier_dir
            / "nanoclustering_g4_8_first_pass_016_objective_barrier_gate_matrix.csv"
        ),
    }


def _local_substrate_rows(local_graph: pd.DataFrame, pair_gates: pd.DataFrame) -> pd.DataFrame:
    graph = local_graph[local_graph["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    gate = pair_gates[pair_gates["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    if len(graph) != 1 or len(gate) != 1:
        raise ValueError("expected exactly one local graph and pair-gate row for local_pair_016")
    graph_row = graph.iloc[0]
    gate_row = gate.iloc[0]
    row = {
        "local_pair_id": PRIMARY_PAIR_ID,
        "object_role_universe_id": str(graph_row["object_role_universe_id"]),
        "branch": str(graph_row["branch"]),
        "left_node_id": int(graph_row["left_node_id"]),
        "right_node_id": int(graph_row["right_node_id"]),
        "pair_scope": str(graph_row["pair_scope"]),
        "counterfactual_class": str(graph_row["counterfactual_class"]),
        "mechanism_label": str(graph_row["mechanism_label"]),
        "direct_edge_weight": float(graph_row["direct_edge_weight"]),
        "direct_positive_at_gamma": bool(_as_bool(graph_row["direct_positive_at_gamma"])),
        "direct_edge_needed_for_input_gamma_positive": bool(
            _as_bool(graph_row["direct_edge_needed_for_input_gamma_positive"])
        ),
        "direct_cpm_delta_q": float(graph_row["direct_cpm_delta_q"]),
        "direct_critical_gamma": float(graph_row["direct_critical_gamma"]),
        "bridge_to_direct_weight_ratio": float(graph_row["bridge_to_direct_weight_ratio"]),
        "bridge_to_input_penalty_ratio": float(graph_row["bridge_to_input_penalty_ratio"]),
        "bridge_to_weighted_degree_floor_ratio": float(
            graph_row["bridge_to_weighted_degree_floor_ratio"]
        ),
        "selected_bridge_count": int(graph_row["selected_bridge_count"]),
        "local_node_count": int(graph_row["local_node_count"]),
        "selected_bridge_node_ids": str(graph_row["selected_bridge_node_ids"]),
        "top_common_neighbors": str(graph_row["top_common_neighbors"]),
        "original_pair_coassigned_share": float(gate_row["original_pair_coassigned_share"]),
        "drop_direct_pair_coassigned_share": float(
            gate_row["drop_direct_pair_coassigned_share"]
        ),
        "drop_bridge_pair_coassigned_share": float(
            gate_row["drop_bridge_pair_coassigned_share"]
        ),
        "drop_direct_and_bridge_pair_coassigned_share": float(
            gate_row["drop_direct_and_bridge_pair_coassigned_share"]
        ),
        "original_distinct_endpoint_count": int(gate_row["original_distinct_endpoint_count"]),
        "original_recurrent_endpoint_count": int(
            gate_row["original_recurrent_endpoint_count"]
        ),
        "original_has_local_switch_signal": bool(
            _as_bool(gate_row["original_has_local_switch_signal"])
        ),
        "gate_class": str(gate_row["gate_class"]),
        "gate_status": str(gate_row["gate_status"]),
        "mechanism_substrate_class": (
            "direct_edge_sensitive_bridge_mass_competition_substrate"
        ),
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "method_status": METHOD_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_status": RUN_STATUS,
    }
    return pd.DataFrame([row])


def _variant_mechanism_rows(variant_summary: pd.DataFrame) -> pd.DataFrame:
    rows = variant_summary[
        variant_summary["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)
    ].copy()
    rows = rows.sort_values("graph_variant", kind="mergesort").reset_index(drop=True)
    expected_roles = {
        "original": "ambiguous_original_source_family_surface",
        "drop_direct_edge": "direct_removed_bridge_split_guard_anchor",
        "drop_bridge_edges": "bridge_removed_pair_coassigned_target_anchor",
        "drop_direct_and_bridge_edges": "both_removed_pair_separated_no_bridge_control",
    }
    rows["variant_mechanism_role"] = rows["graph_variant"].map(expected_roles)
    rows["variant_mechanism_pass"] = rows["variant_mechanism_role"].notna()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    rows["run_status"] = RUN_STATUS
    keep = [
        "local_pair_id",
        "graph_variant",
        "variant_mechanism_role",
        "run_count",
        "start_condition_count",
        "seed_count",
        "distinct_endpoint_count",
        "recurrent_endpoint_count",
        "pair_coassigned_share",
        "mechanism_read_counts",
        "quality_median",
        "cluster_count_median",
        "variant_mechanism_pass",
        "route_execution_status",
        "wall_promotion_status",
        "method_status",
        "claim_boundary",
        "run_status",
    ]
    return rows[keep]


def _fraction_mechanism_rows(
    *,
    persistence_trace: pd.DataFrame,
    reverse_trace: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for direction, trace in [("forward", persistence_trace), ("reverse", reverse_trace)]:
        rows = (
            trace.groupby(
                [
                    "bridge_edge_weight_fraction",
                    "result_endpoint_signature_id",
                    "mechanism_read",
                    "pair_coassigned",
                    "left_bridge_same_cluster_count",
                    "right_bridge_same_cluster_count",
                    "pair_bridge_same_cluster_count",
                ],
                dropna=False,
                sort=True,
            )
            .size()
            .reset_index(name="route_count")
        )
        rows["direction"] = direction
        frames.append(rows)
    rows = pd.concat(frames, ignore_index=True)

    def expected_role(fraction: float) -> str:
        if any(abs(fraction - value) <= EPS for value in TARGET_FRACTIONS):
            return "target_pair_coassigned_without_selected_bridge"
        if any(abs(fraction - value) <= EPS for value in TRANSIENT_FRACTIONS):
            return "transient_pair_separated_single_side_bridge"
        if any(abs(fraction - value) <= EPS for value in SOURCE_FAMILY_FRACTIONS):
            return "source_family_high_bridge_surface"
        return "other"

    def mechanism_pass(row: pd.Series) -> bool:
        role = str(row["expected_mechanism_role"])
        mechanism = str(row["mechanism_read"])
        if role == "target_pair_coassigned_without_selected_bridge":
            return mechanism == "pair_coassigned_without_selected_bridge"
        if role == "transient_pair_separated_single_side_bridge":
            return (
                mechanism == "pair_separated_single_side_bridge"
                and not _as_bool(row["pair_coassigned"])
                and int(row["left_bridge_same_cluster_count"]) == 1
                and int(row["right_bridge_same_cluster_count"]) == 0
                and int(row["pair_bridge_same_cluster_count"]) == 0
            )
        if role == "source_family_high_bridge_surface":
            return mechanism in {
                "pair_coassigned_with_selected_bridge",
                "pair_separated_bridge_split",
            }
        return False

    rows["expected_mechanism_role"] = rows["bridge_edge_weight_fraction"].astype(
        float
    ).map(expected_role)
    rows["fraction_mechanism_pass"] = rows.apply(mechanism_pass, axis=1)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    rows["run_status"] = RUN_STATUS
    return rows[
        [
            "direction",
            "bridge_edge_weight_fraction",
            "expected_mechanism_role",
            "result_endpoint_signature_id",
            "mechanism_read",
            "pair_coassigned",
            "left_bridge_same_cluster_count",
            "right_bridge_same_cluster_count",
            "pair_bridge_same_cluster_count",
            "route_count",
            "fraction_mechanism_pass",
            "route_execution_status",
            "wall_promotion_status",
            "method_status",
            "claim_boundary",
            "run_status",
        ]
    ].sort_values(
        ["direction", "bridge_edge_weight_fraction", "mechanism_read"],
        kind="mergesort",
    )


def _route_mechanism_rows(
    *,
    semantic_route: pd.DataFrame,
    pathway_route: pd.DataFrame,
) -> pd.DataFrame:
    rows = pathway_route[
        [
            "route_key",
            "start_condition",
            "seed",
            "pathway_shape_class",
            "preferred_source_equivalence_status",
            "guard_only_source_family_overlap",
        ]
    ].merge(
        semantic_route[
            [
                "route_key",
                "transient_pair_separated",
                "transient_left_bridge_only",
                "transient_support_equidistant_to_three_anchors",
                "objective_monotone_debt_via_transient",
                "target_persists_after_transient",
                "route_gateway_candidate",
            ]
        ],
        on="route_key",
        how="left",
    )
    bool_cols = [
        "transient_pair_separated",
        "transient_left_bridge_only",
        "transient_support_equidistant_to_three_anchors",
        "objective_monotone_debt_via_transient",
        "target_persists_after_transient",
        "route_gateway_candidate",
    ]
    for column in bool_cols:
        rows[column] = rows[column].map(_as_bool)
    rows["mechanism_route_pass"] = rows[bool_cols].all(axis=1)
    rows["mechanism_route_class"] = rows["pathway_shape_class"].map(
        {
            "bidirectional_source_family_transition_band_guard_caveat": "single_side_bridge_transition_band_guard_caveat",
            "bidirectional_source_family_transition_band_anchor_mismatch": "single_side_bridge_transition_band_anchor_mismatch",
            "bidirectional_source_family_transition_band_strict_source": "single_side_bridge_transition_band_strict_source",
        }
    )
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    rows["run_status"] = RUN_STATUS
    return rows


def _decision_rows(
    *,
    local_rows: pd.DataFrame,
    variant_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    pathway_summary: dict[str, Any],
    objective_summary: dict[str, Any],
) -> pd.DataFrame:
    local = local_rows.iloc[0]
    transient_rows = fraction_rows[
        fraction_rows["expected_mechanism_role"].astype(str).eq(
            "transient_pair_separated_single_side_bridge"
        )
    ]
    target_rows = fraction_rows[
        fraction_rows["expected_mechanism_role"].astype(str).eq(
            "target_pair_coassigned_without_selected_bridge"
        )
    ]
    source_rows = fraction_rows[
        fraction_rows["expected_mechanism_role"].astype(str).eq(
            "source_family_high_bridge_surface"
        )
    ]
    return pd.DataFrame(
        [
            {
                "decision_id": "D1_local_substrate_is_direct_bridge_competition",
                "axis": "local_substrate",
                "observed": {
                    "mechanism_label": local["mechanism_label"],
                    "direct_positive_at_gamma": bool(local["direct_positive_at_gamma"]),
                    "direct_edge_needed_for_input_gamma_positive": bool(
                        local["direct_edge_needed_for_input_gamma_positive"]
                    ),
                    "bridge_to_direct_weight_ratio": float(
                        local["bridge_to_direct_weight_ratio"]
                    ),
                    "selected_bridge_count": int(local["selected_bridge_count"]),
                    "gate_status": local["gate_status"],
                },
                "decision": "016_has_direct_edge_sensitive_bridge_mass_competition_substrate",
                "passes": bool(local["direct_positive_at_gamma"])
                and bool(local["direct_edge_needed_for_input_gamma_positive"])
                and float(local["bridge_to_direct_weight_ratio"]) > 1.0
                and str(local["gate_status"]).startswith("diagnostic_supports"),
                "claim_effect": "names the local mechanism substrate without promoting a wall",
            },
            {
                "decision_id": "D2_counterfactual_variants_separate_mechanism_roles",
                "axis": "variant_roles",
                "observed": variant_rows[
                    [
                        "graph_variant",
                        "variant_mechanism_role",
                        "pair_coassigned_share",
                        "mechanism_read_counts",
                    ]
                ].to_dict("records"),
                "decision": "edge_removal_variants_define_source_target_and_guard_roles",
                "passes": int(variant_rows["variant_mechanism_pass"].map(_as_bool).sum())
                == 4,
                "claim_effect": "anchors the mechanism vocabulary to local counterfactuals",
            },
            {
                "decision_id": "D3_transition_band_is_single_side_bridge_state",
                "axis": "transition_band_mechanism",
                "observed": {
                    "transient_rows": transient_rows[
                        [
                            "direction",
                            "bridge_edge_weight_fraction",
                            "mechanism_read",
                            "route_count",
                            "fraction_mechanism_pass",
                        ]
                    ].to_dict("records"),
                },
                "decision": "transient_band_is_pair_separated_single_side_bridge",
                "passes": bool(transient_rows["fraction_mechanism_pass"].map(_as_bool).all())
                and int(transient_rows["route_count"].sum()) == 288,
                "claim_effect": "identifies the finite band as a bridge-assignment mechanism, not an objective barrier",
            },
            {
                "decision_id": "D4_endpoint_roles_are_mechanistically_distinct",
                "axis": "endpoint_roles",
                "observed": {
                    "target_rows": target_rows[
                        [
                            "direction",
                            "mechanism_read",
                            "route_count",
                            "fraction_mechanism_pass",
                        ]
                    ].to_dict("records"),
                    "source_rows": source_rows[
                        [
                            "direction",
                            "mechanism_read",
                            "route_count",
                            "fraction_mechanism_pass",
                        ]
                    ].to_dict("records"),
                },
                "decision": "target_is_pair_coassigned_without_bridge_and_source_is_high_bridge_surface",
                "passes": bool(target_rows["fraction_mechanism_pass"].map(_as_bool).all())
                and bool(source_rows["fraction_mechanism_pass"].map(_as_bool).all()),
                "claim_effect": "keeps endpoint roles separate from the transient mechanism",
            },
            {
                "decision_id": "D5_route_level_mechanism_recurrence",
                "axis": "route_recurrence",
                "observed": {
                    "route_count": int(len(route_rows)),
                    "mechanism_route_pass_count": int(
                        route_rows["mechanism_route_pass"].map(_as_bool).sum()
                    ),
                    "mechanism_route_class_counts": route_rows[
                        "mechanism_route_class"
                    ].value_counts().to_dict(),
                    "pathway_readout": pathway_summary.get("preferred_pathway_readout"),
                    "objective_profile_class": objective_summary.get(
                        "objective_profile_class"
                    ),
                },
                "decision": "single_side_bridge_transition_band_recurs_in_all_24_routes",
                "passes": int(route_rows["mechanism_route_pass"].map(_as_bool).sum()) == 24,
                "claim_effect": "supports a named 016 mechanism object while keeping generality untested",
            },
            {
                "decision_id": "D6_claim_boundary",
                "axis": "claim_boundary",
                "observed": CLAIM_BOUNDARY,
                "decision": "mechanism_interpretation_only",
                "passes": True,
                "claim_effect": "wall, method, full replay, and quality/cost claims remain closed",
            },
        ]
    )


def _gate_matrix(
    *,
    semantic_gates: pd.DataFrame,
    pathway_gates: pd.DataFrame,
    objective_gates: pd.DataFrame,
    local_rows: pd.DataFrame,
    variant_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
) -> pd.DataFrame:
    transient_rows = fraction_rows[
        fraction_rows["expected_mechanism_role"].astype(str).eq(
            "transient_pair_separated_single_side_bridge"
        )
    ]
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_audits_passed",
                "Did semantic, pathway-shape, and objective/barrier upstream audits pass?",
                {
                    "semantic": semantic_gates["gate_status"].value_counts().to_dict(),
                    "pathway": pathway_gates["gate_status"].value_counts().to_dict(),
                    "objective": objective_gates["gate_status"].value_counts().to_dict(),
                },
                "all upstream gates pass",
                bool(semantic_gates["gate_status"].astype(str).eq("pass").all())
                and bool(pathway_gates["gate_status"].astype(str).eq("pass").all())
                and bool(objective_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_local_substrate_named",
                "Is the local substrate a direct-edge-sensitive bridge-mass competition case?",
                local_rows[
                    [
                        "mechanism_label",
                        "direct_positive_at_gamma",
                        "direct_edge_needed_for_input_gamma_positive",
                        "bridge_to_direct_weight_ratio",
                        "selected_bridge_count",
                        "gate_status",
                    ]
                ].to_dict("records"),
                "direct positive, direct edge needed, bridge mass > direct edge, local switch gate passed",
                bool(local_rows.iloc[0]["direct_positive_at_gamma"])
                and bool(local_rows.iloc[0]["direct_edge_needed_for_input_gamma_positive"])
                and float(local_rows.iloc[0]["bridge_to_direct_weight_ratio"]) > 1.0
                and str(local_rows.iloc[0]["gate_status"]).startswith(
                    "diagnostic_supports"
                ),
            ),
            _gate_row(
                "G3_counterfactual_roles_named",
                "Do local edge-removal variants separate the source, target, and guard roles?",
                variant_rows[
                    [
                        "graph_variant",
                        "variant_mechanism_role",
                        "pair_coassigned_share",
                        "variant_mechanism_pass",
                    ]
                ].to_dict("records"),
                "all four variants map to named roles",
                int(variant_rows["variant_mechanism_pass"].map(_as_bool).sum()) == 4,
            ),
            _gate_row(
                "G4_transition_band_mechanism_consistent",
                "Is the finite transient band consistently pair-separated single-side bridge?",
                transient_rows[
                    [
                        "direction",
                        "bridge_edge_weight_fraction",
                        "mechanism_read",
                        "route_count",
                        "fraction_mechanism_pass",
                    ]
                ].to_dict("records"),
                "12 direction-fraction rows and 288 route states pass single-side bridge predicate",
                len(transient_rows) == 12
                and int(transient_rows["route_count"].sum()) == 288
                and bool(transient_rows["fraction_mechanism_pass"].map(_as_bool).all()),
            ),
            _gate_row(
                "G5_route_recurrence_complete",
                "Does the route-level mechanism recur in all 24 routes?",
                {
                    "route_count": int(len(route_rows)),
                    "mechanism_route_pass_count": int(
                        route_rows["mechanism_route_pass"].map(_as_bool).sum()
                    ),
                    "mechanism_route_class_counts": route_rows[
                        "mechanism_route_class"
                    ].value_counts().to_dict(),
                },
                "24/24 route rows pass mechanism recurrence predicates",
                len(route_rows) == 24
                and int(route_rows["mechanism_route_pass"].map(_as_bool).sum()) == 24,
            ),
            _gate_row(
                "G6_claim_boundaries_closed",
                "Are wall, method, full replay, and quality/cost claims closed?",
                {
                    "decision_passes": int(decision_rows["passes"].map(_as_bool).sum()),
                    "claim_boundary": CLAIM_BOUNDARY,
                },
                "all decisions pass and claim boundary is read-only",
                bool(decision_rows["passes"].map(_as_bool).all()),
            ),
        ]
    )


def _summary(
    *,
    output_dir: Path,
    local_ablation_dir: Path,
    semantic_dir: Path,
    pathway_shape_dir: Path,
    objective_barrier_dir: Path,
    local_rows: pd.DataFrame,
    variant_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    local = local_rows.iloc[0]
    transient_rows = fraction_rows[
        fraction_rows["expected_mechanism_role"].astype(str).eq(
            "transient_pair_separated_single_side_bridge"
        )
    ]
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_mechanism_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "semantic_dir": str(semantic_dir),
        "pathway_shape_dir": str(pathway_shape_dir),
        "objective_barrier_dir": str(objective_barrier_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "mechanism_interpretation_class": (
            "direct_edge_sensitive_bridge_mass_competition_with_single_side_bridge_transition_band"
        ),
        "local_substrate_row_count": int(len(local_rows)),
        "variant_row_count": int(len(variant_rows)),
        "fraction_mechanism_row_count": int(len(fraction_rows)),
        "route_mechanism_row_count": int(len(route_rows)),
        "decision_row_count": int(len(decision_rows)),
        "selected_bridge_count": int(local["selected_bridge_count"]),
        "bridge_to_direct_weight_ratio": float(local["bridge_to_direct_weight_ratio"]),
        "direct_cpm_delta_q": float(local["direct_cpm_delta_q"]),
        "original_pair_coassigned_share": float(local["original_pair_coassigned_share"]),
        "drop_bridge_pair_coassigned_share": float(
            local["drop_bridge_pair_coassigned_share"]
        ),
        "drop_direct_pair_coassigned_share": float(
            local["drop_direct_pair_coassigned_share"]
        ),
        "transient_band_direction_fraction_rows": int(len(transient_rows)),
        "transient_band_route_state_count": int(transient_rows["route_count"].sum()),
        "route_mechanism_pass_count": int(
            route_rows["mechanism_route_pass"].map(_as_bool).sum()
        ),
        "mechanism_route_class_counts": route_rows[
            "mechanism_route_class"
        ].value_counts().to_dict(),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "The 016 source-family transition band has a named local mechanism: "
            "a weak direct-positive pair sits inside a much larger selected-bridge "
            "mass, and bridge-fraction perturbation exposes a finite band where "
            "the pair separates while exactly one selected bridge remains with "
            "the left endpoint. Dropping bridges collapses the pair into the "
            "target-like pair-coassigned-without-bridge state; dropping the "
            "direct edge yields a bridge-split guard/source-family state. This "
            "is a mechanism object for 016 only, not wall, method, full replay, "
            "quality/cost, or generality evidence."
        ),
        "recommended_next_gate": (
            "Move to generalization: test whether analogous direct-edge-sensitive "
            "single-side-bridge transition bands recur in other strict-ready "
            "local pairs, using the fixed mechanism predicates before any broad "
            "threshold/localization sweep."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    path: Path,
    summary: dict[str, Any],
    local_rows: pd.DataFrame,
    variant_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Mechanism Interpretation Audit",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- mechanism_interpretation_class: {summary['mechanism_interpretation_class']}",
        f"- selected_bridge_count: {summary['selected_bridge_count']}",
        f"- bridge_to_direct_weight_ratio: {summary['bridge_to_direct_weight_ratio']}",
        f"- transient_band_route_state_count: {summary['transient_band_route_state_count']}",
        f"- route_mechanism_pass_count: {summary['route_mechanism_pass_count']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Local Substrate",
        "",
        _markdown_table(
            local_rows,
            [
                "mechanism_label",
                "direct_positive_at_gamma",
                "direct_edge_needed_for_input_gamma_positive",
                "bridge_to_direct_weight_ratio",
                "selected_bridge_count",
                "original_pair_coassigned_share",
                "drop_direct_pair_coassigned_share",
                "drop_bridge_pair_coassigned_share",
                "gate_status",
            ],
            max_rows=5,
        ),
        "",
        "## Variant Mechanism Rows",
        "",
        _markdown_table(
            variant_rows,
            [
                "graph_variant",
                "variant_mechanism_role",
                "pair_coassigned_share",
                "mechanism_read_counts",
                "variant_mechanism_pass",
            ],
            max_rows=10,
        ),
        "",
        "## Fraction Mechanism Rows",
        "",
        _markdown_table(
            fraction_rows,
            [
                "direction",
                "bridge_edge_weight_fraction",
                "expected_mechanism_role",
                "mechanism_read",
                "pair_coassigned",
                "left_bridge_same_cluster_count",
                "right_bridge_same_cluster_count",
                "pair_bridge_same_cluster_count",
                "route_count",
                "fraction_mechanism_pass",
            ],
            max_rows=40,
        ),
        "",
        "## Route Mechanism Rows",
        "",
        _markdown_table(
            route_rows,
            [
                "route_key",
                "mechanism_route_class",
                "transient_pair_separated",
                "transient_left_bridge_only",
                "transient_support_equidistant_to_three_anchors",
                "target_persists_after_transient",
                "mechanism_route_pass",
            ],
            max_rows=30,
        ),
        "",
        "## Decisions",
        "",
        _markdown_table(
            decision_rows,
            ["decision_id", "axis", "observed", "decision", "passes", "claim_effect"],
            max_rows=20,
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
        str(summary["interpretation"]),
        "",
        "## Recommended Next Gate",
        "",
        str(summary["recommended_next_gate"]),
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    *,
    output_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    local_rows: pd.DataFrame,
    variant_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(local_rows, output_dir / LOCAL_SUBSTRATE_ROWS_CSV)
    _write_csv(variant_rows, output_dir / VARIANT_MECHANISM_ROWS_CSV)
    _write_csv(fraction_rows, output_dir / FRACTION_MECHANISM_ROWS_CSV)
    _write_csv(route_rows, output_dir / ROUTE_MECHANISM_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
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
        local_rows=local_rows,
        variant_rows=variant_rows,
        fraction_rows=fraction_rows,
        route_rows=route_rows,
        decision_rows=decision_rows,
        gates=gates,
    )


def run_audit(
    *,
    local_ablation_dir: Path = DEFAULT_LOCAL_ABLATION_DIR,
    semantic_dir: Path = DEFAULT_SEMANTIC_DIR,
    persistence_dir: Path = DEFAULT_PERSISTENCE_DIR,
    reverse_dir: Path = DEFAULT_REVERSE_DIR,
    pathway_shape_dir: Path = DEFAULT_PATHWAY_SHAPE_DIR,
    objective_barrier_dir: Path = DEFAULT_OBJECTIVE_BARRIER_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    context = _load_context(
        local_ablation_dir=local_ablation_dir,
        semantic_dir=semantic_dir,
        persistence_dir=persistence_dir,
        reverse_dir=reverse_dir,
        pathway_shape_dir=pathway_shape_dir,
        objective_barrier_dir=objective_barrier_dir,
    )
    local_rows = _local_substrate_rows(context["local_graph"], context["pair_gates"])
    variant_rows = _variant_mechanism_rows(context["variant_summary"])
    fraction_rows = _fraction_mechanism_rows(
        persistence_trace=context["persistence_trace"],
        reverse_trace=context["reverse_trace"],
    )
    route_rows = _route_mechanism_rows(
        semantic_route=context["semantic_route"],
        pathway_route=context["pathway_route"],
    )
    decision_rows = _decision_rows(
        local_rows=local_rows,
        variant_rows=variant_rows,
        fraction_rows=fraction_rows,
        route_rows=route_rows,
        pathway_summary=context["pathway_summary"],
        objective_summary=context["objective_summary"],
    )
    gates = _gate_matrix(
        semantic_gates=context["semantic_gates"],
        pathway_gates=context["pathway_gates"],
        objective_gates=context["objective_gates"],
        local_rows=local_rows,
        variant_rows=variant_rows,
        fraction_rows=fraction_rows,
        route_rows=route_rows,
        decision_rows=decision_rows,
    )
    summary = _summary(
        output_dir=output_dir,
        local_ablation_dir=local_ablation_dir,
        semantic_dir=semantic_dir,
        pathway_shape_dir=pathway_shape_dir,
        objective_barrier_dir=objective_barrier_dir,
        local_rows=local_rows,
        variant_rows=variant_rows,
        fraction_rows=fraction_rows,
        route_rows=route_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_mechanism_config.v1",
        "local_ablation_dir": str(local_ablation_dir),
        "semantic_dir": str(semantic_dir),
        "persistence_dir": str(persistence_dir),
        "reverse_dir": str(reverse_dir),
        "pathway_shape_dir": str(pathway_shape_dir),
        "objective_barrier_dir": str(objective_barrier_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "method_status": METHOD_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_status": RUN_STATUS,
    }
    _write_outputs(
        output_dir=output_dir,
        config=config,
        summary=summary,
        local_rows=local_rows,
        variant_rows=variant_rows,
        fraction_rows=fraction_rows,
        route_rows=route_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local_pair_016 mechanism interpretation.",
    )
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--semantic-dir", type=Path, default=DEFAULT_SEMANTIC_DIR)
    parser.add_argument("--persistence-dir", type=Path, default=DEFAULT_PERSISTENCE_DIR)
    parser.add_argument("--reverse-dir", type=Path, default=DEFAULT_REVERSE_DIR)
    parser.add_argument("--pathway-shape-dir", type=Path, default=DEFAULT_PATHWAY_SHAPE_DIR)
    parser.add_argument(
        "--objective-barrier-dir",
        type=Path,
        default=DEFAULT_OBJECTIVE_BARRIER_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_audit(
        local_ablation_dir=args.local_ablation_dir,
        semantic_dir=args.semantic_dir,
        persistence_dir=args.persistence_dir,
        reverse_dir=args.reverse_dir,
        pathway_shape_dir=args.pathway_shape_dir,
        objective_barrier_dir=args.objective_barrier_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
