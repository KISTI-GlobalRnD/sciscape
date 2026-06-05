#!/usr/bin/env python3
"""Test a frozen bridge-context-release handle on independent synthetic variants.

This G4.3 probe freezes the G4.2 mechanism cue as a predeclared handle:
release bridge nodes from pair-node clusters into same-side host context
without merging the L/R pair. It then tests that handle on a small, fixed panel
of independent controlled variants and matched controls.

The handle does not read a target endpoint signature. The probe first freezes
ordinary Leiden+CPM baseline endpoints for each variant, then applies only
source replay, pair-only merge, and the frozen bridge-context-release handle
from separated source endpoints.

It is a synthetic handle-generalization diagnostic only. It does not promote
basin walls, prove pathways, compare a full method, replay NanoClustering, make
quality/cost claims, or claim an algorithm.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.clustering.partitioning import partition_class
from sciscape.clustering.runner import LeidenRunner

from analyze_leiden_cpm_variable_pair_synthetic_route_trace_g4_1_audit import (
    _changed_node_count,
    _coassociation_distance,
    _groups_by_node,
    _groups_to_labels,
    _labels_to_membership,
)
from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    START_CONDITIONS,
    SyntheticCase,
    _build_graph,
    _canonical_groups,
    _competition_case,
    _edges_for_variant,
    _initial_membership,
    _json_safe,
    _mechanism_read,
    _signature_id,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_3_handle_generalization_v1_20260603"
)

PANEL_CASES_CSV = "variable_pair_synthetic_g4_3_panel_cases.csv"
GRAPH_MANIFEST_CSV = "variable_pair_synthetic_g4_3_graph_manifest.csv"
GRAPH_EDGES_CSV = "variable_pair_synthetic_g4_3_graph_edges.csv"
BASELINE_RUNS_CSV = "variable_pair_synthetic_g4_3_baseline_runs.csv"
ENDPOINT_SUMMARY_CSV = "variable_pair_synthetic_g4_3_endpoint_summary.csv"
HANDLE_RUNS_CSV = "variable_pair_synthetic_g4_3_handle_runs.csv"
HANDLE_POLICY_SUMMARY_CSV = "variable_pair_synthetic_g4_3_handle_policy_summary.csv"
VARIANT_GATE_ROWS_CSV = "variable_pair_synthetic_g4_3_variant_gate_rows.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_3_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_3_config.json"
REPORT_MD = "variable_pair_synthetic_g4_3_report.md"

HANDLE_POLICIES = (
    "source_replay",
    "pair_relation_only",
    "bridge_context_release_without_pair_merge",
)
CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.3 handle-generalization diagnostic only; a "
    "predeclared bridge-context-release handle is tested on a fixed synthetic "
    "variant/control panel without target endpoint reconstruction. No basin-wall "
    "promotion, no pathway claim, no full-method comparison, no full "
    "NanoClustering replay, no quality/cost claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "executed_g4_3_frozen_handle_generalization_probe"
WALL_PROMOTION_STATUS = "not_promoted_handle_generalization_only"
METHOD_STATUS = "frozen_handle_probe_not_method_claim"


@dataclass(frozen=True)
class PanelCase:
    case_id: str
    panel_role: str
    expected_gate: str
    direct_weight: float
    pair_bridge_weight: float
    bridge_host_weight: float
    host_clique_weight: float
    pair_node_size: int = 1
    note: str = ""


PANEL_CASES: tuple[PanelCase, ...] = (
    PanelCase(
        case_id="positive_direct_low_104",
        panel_role="positive_holdout",
        expected_gate="bridge_release_robust_pair_coassignment",
        direct_weight=1.04,
        pair_bridge_weight=1.35,
        bridge_host_weight=1.45,
        host_clique_weight=1.25,
        note="Direct weight below the G3 rare case; context threshold unchanged.",
    ),
    PanelCase(
        case_id="positive_direct_high_112",
        panel_role="positive_holdout",
        expected_gate="bridge_release_robust_pair_coassignment",
        direct_weight=1.12,
        pair_bridge_weight=1.35,
        bridge_host_weight=1.45,
        host_clique_weight=1.25,
        note="Direct weight above the G3 rare case; context threshold unchanged.",
    ),
    PanelCase(
        case_id="positive_host_low_120",
        panel_role="positive_holdout",
        expected_gate="bridge_release_robust_pair_coassignment",
        direct_weight=1.08,
        pair_bridge_weight=1.35,
        bridge_host_weight=1.45,
        host_clique_weight=1.20,
        note="Host-clique weight below the G3 rare case.",
    ),
    PanelCase(
        case_id="positive_host_high_130",
        panel_role="positive_holdout",
        expected_gate="bridge_release_robust_pair_coassignment",
        direct_weight=1.08,
        pair_bridge_weight=1.35,
        bridge_host_weight=1.45,
        host_clique_weight=1.30,
        note="Host-clique weight above the G3 rare case.",
    ),
    PanelCase(
        case_id="control_context_below_threshold_142",
        panel_role="matched_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.08,
        pair_bridge_weight=1.35,
        bridge_host_weight=1.42,
        host_clique_weight=1.25,
        note="Same pair/direct scale, lower bridge-host context.",
    ),
    PanelCase(
        case_id="control_context_below_threshold_143",
        panel_role="matched_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.08,
        pair_bridge_weight=1.35,
        bridge_host_weight=1.43,
        host_clique_weight=1.25,
        note="Near-threshold bridge-host context control.",
    ),
    PanelCase(
        case_id="control_pair_bridge_high_140",
        panel_role="matched_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.08,
        pair_bridge_weight=1.40,
        bridge_host_weight=1.45,
        host_clique_weight=1.25,
        note="Pair-bridge weight above the G3 rare case.",
    ),
    PanelCase(
        case_id="control_no_direct_support_055",
        panel_role="negative_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=0.55,
        pair_bridge_weight=1.35,
        bridge_host_weight=1.45,
        host_clique_weight=1.25,
        note="Direct-pair support removed while context remains strong.",
    ),
    PanelCase(
        case_id="control_weak_context_095",
        panel_role="negative_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.08,
        pair_bridge_weight=1.35,
        bridge_host_weight=0.95,
        host_clique_weight=1.25,
        note="Bridge-host context is too weak for release to anchor.",
    ),
)


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _panel_case_to_synthetic(case: PanelCase) -> SyntheticCase:
    return _competition_case(
        design_family=case.case_id,
        synthetic_demo_role=case.panel_role,
        expected_signature=case.expected_gate,
        direct_weight=case.direct_weight,
        pair_bridge_weight=case.pair_bridge_weight,
        bridge_host_weight=case.bridge_host_weight,
        host_clique_weight=case.host_clique_weight,
        gamma=1.0,
        pair_node_size=case.pair_node_size,
    )


def _quality_for_membership(
    *,
    graph,
    gamma: float,
    membership: list[int],
    node_sizes: tuple[int, ...],
) -> tuple[float, int]:
    partition_cls = partition_class("cpm")
    weights = graph.es["weight"] if "weight" in graph.es.attributes() else None
    partition = partition_cls(
        graph,
        weights=weights,
        resolution_parameter=float(gamma),
        initial_membership=list(membership),
        node_sizes=list(node_sizes),
    )
    return float(partition.quality()), int(len(partition))


def _host_label(labels: dict[str, int], prefix: str) -> int | None:
    host_nodes = sorted(node for node in labels if node.startswith(prefix))
    if not host_nodes:
        return None
    return int(labels[host_nodes[0]])


def _release_bridge_context_without_pair_merge(
    *,
    case: SyntheticCase,
    source_membership: list[int],
) -> tuple[list[int], dict[str, Any]]:
    labels = {
        str(node): int(label)
        for node, label in zip(case.nodes, source_membership, strict=True)
    }
    groups = _groups_by_node(case.nodes, source_membership)
    left_host = _host_label(labels, "la")
    right_host = _host_label(labels, "ra")
    released: list[str] = []
    skipped: list[str] = []
    for bridge in case.bridge_nodes:
        bridge = str(bridge)
        group = groups[bridge]
        if bridge.startswith("lb") and "L" in group and "R" not in group:
            if left_host is None:
                skipped.append(bridge)
            else:
                labels[bridge] = left_host
                released.append(bridge)
        elif bridge.startswith("rb") and "R" in group and "L" not in group:
            if right_host is None:
                skipped.append(bridge)
            else:
                labels[bridge] = right_host
                released.append(bridge)
        else:
            skipped.append(bridge)
    initial_membership = _labels_to_membership(case.nodes, labels)
    changed_nodes = _changed_node_count(
        case.nodes,
        source_membership,
        initial_membership,
    )
    source_read = _mechanism_read(case, source_membership)
    initial_read = _mechanism_read(case, initial_membership)
    return initial_membership, {
        "released_bridge_nodes": ";".join(sorted(released)),
        "skipped_bridge_nodes": ";".join(sorted(skipped)),
        "released_bridge_count": int(len(released)),
        "changed_nodes_vs_source": int(changed_nodes),
        "source_pair_coassigned": bool(source_read["pair_coassigned"]),
        "initial_pair_coassigned": bool(initial_read["pair_coassigned"]),
        "initial_keeps_pair_relation": bool(
            source_read["pair_coassigned"] == initial_read["pair_coassigned"]
        ),
    }


def _pair_relation_only(
    *,
    case: SyntheticCase,
    source_membership: list[int],
) -> tuple[list[int], dict[str, Any]]:
    labels = {
        str(node): int(label)
        for node, label in zip(case.nodes, source_membership, strict=True)
    }
    labels["R"] = labels["L"]
    initial_membership = _labels_to_membership(case.nodes, labels)
    source_read = _mechanism_read(case, source_membership)
    initial_read = _mechanism_read(case, initial_membership)
    return initial_membership, {
        "released_bridge_nodes": "",
        "skipped_bridge_nodes": "",
        "released_bridge_count": 0,
        "changed_nodes_vs_source": int(
            _changed_node_count(case.nodes, source_membership, initial_membership)
        ),
        "source_pair_coassigned": bool(source_read["pair_coassigned"]),
        "initial_pair_coassigned": bool(initial_read["pair_coassigned"]),
        "initial_keeps_pair_relation": bool(
            source_read["pair_coassigned"] == initial_read["pair_coassigned"]
        ),
    }


def _initial_for_policy(
    *,
    case: SyntheticCase,
    source_membership: list[int],
    policy: str,
) -> tuple[list[int], dict[str, Any]]:
    source_read = _mechanism_read(case, source_membership)
    if policy == "source_replay":
        return list(source_membership), {
            "released_bridge_nodes": "",
            "skipped_bridge_nodes": "",
            "released_bridge_count": 0,
            "changed_nodes_vs_source": 0,
            "source_pair_coassigned": bool(source_read["pair_coassigned"]),
            "initial_pair_coassigned": bool(source_read["pair_coassigned"]),
            "initial_keeps_pair_relation": True,
        }
    if policy == "pair_relation_only":
        return _pair_relation_only(case=case, source_membership=source_membership)
    if policy == "bridge_context_release_without_pair_merge":
        return _release_bridge_context_without_pair_merge(
            case=case,
            source_membership=source_membership,
        )
    raise ValueError(f"unknown handle policy: {policy}")


def _membership_from_signature(case: SyntheticCase, endpoint_signature: str) -> list[int]:
    labels = _groups_to_labels(case.nodes, endpoint_signature)
    return _labels_to_membership(case.nodes, labels)


def _graph_manifest_and_edges(
    cases: list[SyntheticCase],
    panel_cases: tuple[PanelCase, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel_lookup = {case.case_id: case for case in panel_cases}
    panel_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for case in cases:
        panel = panel_lookup[case.design_family]
        panel_rows.append(
            {
                "case_id": panel.case_id,
                "panel_role": panel.panel_role,
                "expected_gate": panel.expected_gate,
                "direct_weight": float(panel.direct_weight),
                "pair_bridge_weight": float(panel.pair_bridge_weight),
                "bridge_host_weight": float(panel.bridge_host_weight),
                "host_clique_weight": float(panel.host_clique_weight),
                "pair_node_size": int(panel.pair_node_size),
                "note": panel.note,
            }
        )
        graph = _build_graph(case.nodes, _edges_for_variant(case, "original"))
        graph_rows.append(
            {
                "case_id": case.design_family,
                "panel_role": panel.panel_role,
                "expected_gate": panel.expected_gate,
                "gamma": float(case.gamma),
                "node_count": int(graph.vcount()),
                "edge_count": int(graph.ecount()),
                "edge_weight_sum": float(sum(graph.es["weight"])),
                "bridge_nodes": ";".join(case.bridge_nodes),
                "node_sizes": ";".join(
                    f"{node}:{size}"
                    for node, size in zip(case.nodes, case.node_sizes, strict=True)
                ),
            }
        )
        for edge in _edges_for_variant(case, "original"):
            edge_rows.append(
                {
                    "case_id": case.design_family,
                    "panel_role": panel.panel_role,
                    "expected_gate": panel.expected_gate,
                    "source": edge.left,
                    "target": edge.right,
                    "weight": float(edge.weight),
                    "edge_type": edge.edge_type,
                }
            )
    return (
        _claim_columns(pd.DataFrame(panel_rows)),
        _claim_columns(pd.DataFrame(graph_rows)),
        _claim_columns(pd.DataFrame(edge_rows)),
    )


def _run_baseline(
    *,
    cases: list[SyntheticCase],
    panel_cases: tuple[PanelCase, ...],
    seeds: int,
    n_iterations: int,
) -> pd.DataFrame:
    panel_lookup = {case.case_id: case for case in panel_cases}
    rows: list[dict[str, Any]] = []
    for case in cases:
        panel = panel_lookup[case.design_family]
        graph = _build_graph(case.nodes, _edges_for_variant(case, "original"))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        for start_condition in START_CONDITIONS:
            initial = _initial_membership(case, start_condition)
            for seed in range(int(seeds)):
                result = runner.run(
                    case.gamma,
                    seed=int(seed),
                    initial_membership=initial,
                    node_sizes=case.node_sizes,
                )
                membership = list(map(int, result.membership))
                groups = _canonical_groups(case.nodes, membership)
                read = _mechanism_read(case, membership)
                rows.append(
                    {
                        "case_id": case.design_family,
                        "panel_role": panel.panel_role,
                        "expected_gate": panel.expected_gate,
                        "start_condition": start_condition,
                        "seed": int(seed),
                        "n_iterations": int(n_iterations),
                        "gamma": float(case.gamma),
                        "endpoint_signature_id": _signature_id(groups),
                        "endpoint_signature": json.dumps(groups, sort_keys=True),
                        "pair_coassigned": bool(read["pair_coassigned"]),
                        "mechanism_read": str(read["mechanism_read"]),
                        "left_bridge_same_cluster_count": int(
                            read["left_bridge_same_cluster_count"]
                        ),
                        "right_bridge_same_cluster_count": int(
                            read["right_bridge_same_cluster_count"]
                        ),
                        "quality": float(result.quality),
                        "cluster_count": int(result.cluster_count),
                    }
                )
    return _claim_columns(pd.DataFrame(rows))


def _endpoint_summary(baseline_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "case_id",
        "panel_role",
        "expected_gate",
        "endpoint_signature_id",
        "endpoint_signature",
        "pair_coassigned",
        "mechanism_read",
    ]
    for keys, group in baseline_runs.groupby(group_cols, sort=True):
        key = dict(zip(group_cols, keys, strict=True))
        rows.append(
            {
                **key,
                "endpoint_run_count": int(len(group)),
                "endpoint_run_share_within_case": float(
                    len(group)
                    / len(baseline_runs[baseline_runs["case_id"].eq(key["case_id"])])
                ),
                "start_condition_count": int(group["start_condition"].nunique()),
                "seed_count": int(group["seed"].nunique()),
                "quality_min": float(group["quality"].min()),
                "quality_median": float(group["quality"].median()),
                "quality_max": float(group["quality"].max()),
                "cluster_count_median": float(group["cluster_count"].median()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        [
            "case_id",
            "pair_coassigned",
            "endpoint_run_count",
            "endpoint_signature_id",
        ],
        ascending=[True, False, False, True],
        kind="stable",
    )
    summary["endpoint_rank_within_case"] = (
        summary.groupby("case_id").cumcount() + 1
    )
    return _claim_columns(summary)


def _endpoint_lookup(endpoint_summary: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["case_id"]), str(row["endpoint_signature_id"])): row
        for row in endpoint_summary.to_dict("records")
    }


def _trace_outcome(
    *,
    case_id: str,
    result_signature_id: str,
    source_signature_id: str,
    endpoint_lookup: dict[tuple[str, str], dict[str, Any]],
) -> str:
    if result_signature_id == source_signature_id:
        return "bounces_to_source"
    matched = endpoint_lookup.get((case_id, result_signature_id))
    if matched is None:
        return "unknown_pair_coassigned"  # corrected below by caller if needed
    if bool(matched["pair_coassigned"]):
        return "collapses_to_known_coassigned_endpoint"
    return "collapses_to_other_known_separated_endpoint"


def _run_handles(
    *,
    cases: list[SyntheticCase],
    panel_cases: tuple[PanelCase, ...],
    endpoint_summary: pd.DataFrame,
    seeds: int,
    n_iterations: int,
) -> pd.DataFrame:
    panel_lookup = {case.case_id: case for case in panel_cases}
    endpoints_by_case = {
        case_id: group.copy()
        for case_id, group in endpoint_summary.groupby("case_id", sort=False)
    }
    endpoint_lookup = _endpoint_lookup(endpoint_summary)
    rows: list[dict[str, Any]] = []
    for case in cases:
        panel = panel_lookup[case.design_family]
        graph = _build_graph(case.nodes, _edges_for_variant(case, "original"))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        endpoints = endpoints_by_case[case.design_family]
        source_endpoints = endpoints[~endpoints["pair_coassigned"].astype(bool)]
        for source in source_endpoints.to_dict("records"):
            source_membership = _membership_from_signature(
                case,
                str(source["endpoint_signature"]),
            )
            source_quality, source_cluster_count = _quality_for_membership(
                graph=graph,
                gamma=case.gamma,
                membership=source_membership,
                node_sizes=case.node_sizes,
            )
            for policy in HANDLE_POLICIES:
                initial, initial_meta = _initial_for_policy(
                    case=case,
                    source_membership=source_membership,
                    policy=policy,
                )
                initial_quality, initial_cluster_count = _quality_for_membership(
                    graph=graph,
                    gamma=case.gamma,
                    membership=initial,
                    node_sizes=case.node_sizes,
                )
                handle_eligible = (
                    policy == "bridge_context_release_without_pair_merge"
                    and bool(initial_meta["initial_keeps_pair_relation"])
                    and not bool(initial_meta["source_pair_coassigned"])
                    and int(initial_meta["released_bridge_count"]) > 0
                    and int(initial_meta["changed_nodes_vs_source"]) > 0
                )
                for seed in range(int(seeds)):
                    result = runner.run(
                        case.gamma,
                        seed=int(seed),
                        initial_membership=initial,
                        node_sizes=case.node_sizes,
                    )
                    membership = list(map(int, result.membership))
                    groups = _canonical_groups(case.nodes, membership)
                    result_signature_id = _signature_id(groups)
                    read = _mechanism_read(case, membership)
                    outcome = _trace_outcome(
                        case_id=case.design_family,
                        result_signature_id=result_signature_id,
                        source_signature_id=str(source["endpoint_signature_id"]),
                        endpoint_lookup=endpoint_lookup,
                    )
                    if endpoint_lookup.get((case.design_family, result_signature_id)) is None:
                        outcome = (
                            "unknown_pair_coassigned"
                            if bool(read["pair_coassigned"])
                            else "unknown_pair_separated"
                        )
                    rows.append(
                        {
                            "case_id": case.design_family,
                            "panel_role": panel.panel_role,
                            "expected_gate": panel.expected_gate,
                            "source_endpoint_signature_id": str(
                                source["endpoint_signature_id"]
                            ),
                            "source_endpoint_run_count": int(
                                source["endpoint_run_count"]
                            ),
                            "source_endpoint_run_share_within_case": float(
                                source["endpoint_run_share_within_case"]
                            ),
                            "source_quality": float(source_quality),
                            "source_cluster_count": int(source_cluster_count),
                            "handle_policy": policy,
                            "handle_eligible": bool(handle_eligible),
                            "trace_seed": int(seed),
                            "n_iterations": int(n_iterations),
                            "initial_quality": float(initial_quality),
                            "initial_cluster_count": int(initial_cluster_count),
                            "initial_quality_delta_vs_source": float(
                                initial_quality - source_quality
                            ),
                            "initial_coassoc_distance_vs_source": int(
                                _coassociation_distance(source_membership, initial)
                            ),
                            "result_endpoint_signature_id": result_signature_id,
                            "trace_outcome": outcome,
                            "result_pair_coassigned": bool(read["pair_coassigned"]),
                            "result_mechanism_read": str(read["mechanism_read"]),
                            "result_quality": float(result.quality),
                            "result_quality_delta_vs_source": float(
                                result.quality - source_quality
                            ),
                            "result_quality_delta_vs_initial": float(
                                result.quality - initial_quality
                            ),
                            "result_cluster_count": int(result.cluster_count),
                            "result_endpoint_signature": json.dumps(
                                groups,
                                sort_keys=True,
                            ),
                            **initial_meta,
                        }
                    )
    return _claim_columns(pd.DataFrame(rows))


def _classify_policy(group: pd.DataFrame) -> str:
    pair_rate = float(group["result_pair_coassigned"].mean())
    source_rate = float(group["trace_outcome"].eq("bounces_to_source").mean())
    known_coassigned_rate = float(
        group["trace_outcome"].eq("collapses_to_known_coassigned_endpoint").mean()
    )
    eligible = bool(group["handle_eligible"].iloc[0])
    policy = str(group["handle_policy"].iloc[0])
    if policy == "source_replay":
        return "source_replay_control"
    if not eligible and policy == "bridge_context_release_without_pair_merge":
        return "handle_not_eligible"
    if pair_rate >= 0.8:
        if known_coassigned_rate >= 0.8:
            return "robust_known_coassigned_endpoint"
        return "robust_pair_coassignment"
    if source_rate >= 0.8:
        return "robust_source_bounce"
    return "mixed_or_partial_pair_coassignment"


def _handle_policy_summary(handle_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "case_id",
        "panel_role",
        "expected_gate",
        "source_endpoint_signature_id",
        "handle_policy",
        "handle_eligible",
        "released_bridge_nodes",
        "released_bridge_count",
        "changed_nodes_vs_source",
        "source_pair_coassigned",
        "initial_pair_coassigned",
        "initial_keeps_pair_relation",
        "initial_quality",
        "initial_cluster_count",
        "initial_quality_delta_vs_source",
        "initial_coassoc_distance_vs_source",
    ]
    for keys, group in handle_runs.groupby(group_cols, sort=True):
        key = dict(zip(group_cols, keys, strict=True))
        rows.append(
            {
                **key,
                "run_count": int(len(group)),
                "pair_coassigned_count": int(group["result_pair_coassigned"].sum()),
                "pair_coassigned_rate": float(group["result_pair_coassigned"].mean()),
                "source_bounce_count": int(
                    group["trace_outcome"].eq("bounces_to_source").sum()
                ),
                "source_bounce_rate": float(
                    group["trace_outcome"].eq("bounces_to_source").mean()
                ),
                "known_coassigned_endpoint_count": int(
                    group["trace_outcome"].eq(
                        "collapses_to_known_coassigned_endpoint"
                    ).sum()
                ),
                "known_coassigned_endpoint_rate": float(
                    group["trace_outcome"].eq(
                        "collapses_to_known_coassigned_endpoint"
                    ).mean()
                ),
                "distinct_result_endpoint_count": int(
                    group["result_endpoint_signature_id"].nunique()
                ),
                "handle_policy_class": _classify_policy(group),
                "result_quality_min": float(group["result_quality"].min()),
                "result_quality_median": float(group["result_quality"].median()),
                "result_quality_max": float(group["result_quality"].max()),
                "result_quality_delta_vs_source_median": float(
                    group["result_quality_delta_vs_source"].median()
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _variant_gate_rows(
    *,
    endpoint_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    panel_cases: tuple[PanelCase, ...],
) -> pd.DataFrame:
    panel_lookup = {case.case_id: case for case in panel_cases}
    rows: list[dict[str, Any]] = []
    for case_id, endpoints in endpoint_summary.groupby("case_id", sort=True):
        panel = panel_lookup[str(case_id)]
        policies = policy_summary[policy_summary["case_id"].eq(case_id)]
        bridge = policies[
            policies["handle_policy"].eq(
                "bridge_context_release_without_pair_merge"
            )
        ]
        pair_only = policies[policies["handle_policy"].eq("pair_relation_only")]
        eligible_bridge = bridge[bridge["handle_eligible"].astype(bool)]
        robust_bridge = eligible_bridge[
            eligible_bridge["pair_coassigned_rate"] >= 0.8
        ]
        robust_pair_only = pair_only[pair_only["pair_coassigned_rate"] >= 0.8]
        endpoint_pair_share = float(
            (
                endpoints["pair_coassigned"].astype(bool)
                * endpoints["endpoint_run_count"].astype(int)
            ).sum()
            / endpoints["endpoint_run_count"].astype(int).sum()
        )
        if panel.expected_gate == "bridge_release_robust_pair_coassignment":
            passed = (
                len(robust_bridge) > 0
                and len(robust_bridge) == len(eligible_bridge)
                and len(robust_pair_only) == 0
                and bool(endpoints["pair_coassigned"].astype(bool).any())
                and bool((~endpoints["pair_coassigned"].astype(bool)).any())
            )
        else:
            passed = len(robust_bridge) == 0
        rows.append(
            {
                "case_id": str(case_id),
                "panel_role": panel.panel_role,
                "expected_gate": panel.expected_gate,
                "endpoint_count": int(len(endpoints)),
                "separated_endpoint_count": int(
                    (~endpoints["pair_coassigned"].astype(bool)).sum()
                ),
                "coassigned_endpoint_count": int(
                    endpoints["pair_coassigned"].astype(bool).sum()
                ),
                "baseline_pair_coassigned_run_share": endpoint_pair_share,
                "bridge_handle_eligible_source_count": int(len(eligible_bridge)),
                "bridge_handle_robust_pair_coassignment_count": int(
                    len(robust_bridge)
                ),
                "pair_relation_only_robust_pair_coassignment_count": int(
                    len(robust_pair_only)
                ),
                "bridge_handle_pair_rate_min": float(
                    eligible_bridge["pair_coassigned_rate"].min()
                )
                if not eligible_bridge.empty
                else 0.0,
                "bridge_handle_pair_rate_median": float(
                    eligible_bridge["pair_coassigned_rate"].median()
                )
                if not eligible_bridge.empty
                else 0.0,
                "bridge_handle_pair_rate_max": float(
                    eligible_bridge["pair_coassigned_rate"].max()
                )
                if not eligible_bridge.empty
                else 0.0,
                "gate_passed": bool(passed),
                "g4_3_gate_status": (
                    "expected_handle_behavior_reproduced"
                    if passed
                    else "expected_handle_behavior_not_reproduced"
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    output_dir: Path,
    baseline_runs: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
    handle_runs: pd.DataFrame,
    policy_summary: pd.DataFrame,
    gate_rows: pd.DataFrame,
) -> dict[str, Any]:
    positive = gate_rows[gate_rows["panel_role"].eq("positive_holdout")]
    controls = gate_rows[~gate_rows["panel_role"].eq("positive_holdout")]
    return {
        "schema": "variable_pair_synthetic_g4_3_handle_generalization_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "output_dir": str(output_dir),
        "panel_case_count": int(gate_rows["case_id"].nunique()),
        "positive_case_count": int(positive["case_id"].nunique()),
        "control_case_count": int(controls["case_id"].nunique()),
        "baseline_run_count": int(len(baseline_runs)),
        "endpoint_count": int(len(endpoint_summary)),
        "handle_run_count": int(len(handle_runs)),
        "handle_policy_row_count": int(len(policy_summary)),
        "gate_status_counts": gate_rows["g4_3_gate_status"].value_counts().to_dict(),
        "positive_pass_count": int(positive["gate_passed"].astype(bool).sum()),
        "positive_fail_count": int((~positive["gate_passed"].astype(bool)).sum()),
        "control_pass_count": int(controls["gate_passed"].astype(bool).sum()),
        "control_fail_count": int((~controls["gate_passed"].astype(bool)).sum()),
        "policy_class_counts": policy_summary[
            "handle_policy_class"
        ].value_counts().to_dict(),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    gate_rows: pd.DataFrame,
    policy_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.3 Handle Generalization Probe",
        "",
        f"- status: `{summary['status']}`",
        f"- panel_case_count: {summary['panel_case_count']}",
        f"- positive_pass_count: {summary['positive_pass_count']}",
        f"- positive_fail_count: {summary['positive_fail_count']}",
        f"- control_pass_count: {summary['control_pass_count']}",
        f"- control_fail_count: {summary['control_fail_count']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- policy_class_counts: {summary['policy_class_counts']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Variant Gates",
    ]
    for row in gate_rows.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id} ({row.panel_role}): {row.g4_3_gate_status}; "
            f"endpoints={row.endpoint_count}, "
            f"baseline_pair_share={row.baseline_pair_coassigned_run_share:.3f}, "
            f"eligible={row.bridge_handle_eligible_source_count}, "
            f"robust_bridge={row.bridge_handle_robust_pair_coassignment_count}, "
            f"robust_pair_only={row.pair_relation_only_robust_pair_coassignment_count}, "
            f"bridge_rate_median={row.bridge_handle_pair_rate_median:.3f}"
        )
    lines.extend(["", "## Frozen Handle Rows"])
    bridge = policy_summary[
        policy_summary["handle_policy"].eq(
            "bridge_context_release_without_pair_merge"
        )
    ]
    for row in bridge.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id} {row.source_endpoint_signature_id}: "
            f"{row.handle_policy_class}, "
            f"eligible={row.handle_eligible}, "
            f"released={row.released_bridge_count}, "
            f"pair_rate={row.pair_coassigned_rate:.3f}, "
            f"initial_delta={row.initial_quality_delta_vs_source:.6g}, "
            f"result_delta_median={row.result_quality_delta_vs_source_median:.6g}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This probe freezes one handle class and tests it on a fixed "
                "synthetic panel. It does not compare a full method, promote a "
                "wall, or claim downstream value."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_cases = PANEL_CASES
    cases = [_panel_case_to_synthetic(case) for case in panel_cases]
    panel_rows, graph_manifest, graph_edges = _graph_manifest_and_edges(
        cases,
        panel_cases,
    )
    baseline_runs = _run_baseline(
        cases=cases,
        panel_cases=panel_cases,
        seeds=int(args.baseline_seeds),
        n_iterations=int(args.n_iterations),
    )
    endpoint_summary = _endpoint_summary(baseline_runs)
    handle_runs = _run_handles(
        cases=cases,
        panel_cases=panel_cases,
        endpoint_summary=endpoint_summary,
        seeds=int(args.handle_seeds),
        n_iterations=int(args.n_iterations),
    )
    policy_summary = _handle_policy_summary(handle_runs)
    gate_rows = _variant_gate_rows(
        endpoint_summary=endpoint_summary,
        policy_summary=policy_summary,
        panel_cases=panel_cases,
    )
    _write_csv(panel_rows, output_dir / PANEL_CASES_CSV)
    _write_csv(graph_manifest, output_dir / GRAPH_MANIFEST_CSV)
    _write_csv(graph_edges, output_dir / GRAPH_EDGES_CSV)
    _write_csv(baseline_runs, output_dir / BASELINE_RUNS_CSV)
    _write_csv(endpoint_summary, output_dir / ENDPOINT_SUMMARY_CSV)
    _write_csv(handle_runs, output_dir / HANDLE_RUNS_CSV)
    _write_csv(policy_summary, output_dir / HANDLE_POLICY_SUMMARY_CSV)
    _write_csv(gate_rows, output_dir / VARIANT_GATE_ROWS_CSV)
    summary = _summary(
        output_dir=output_dir,
        baseline_runs=baseline_runs,
        endpoint_summary=endpoint_summary,
        handle_runs=handle_runs,
        policy_summary=policy_summary,
        gate_rows=gate_rows,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_3_handle_generalization_config.v1",
        "output_dir": str(output_dir),
        "panel_cases": [case.__dict__ for case in panel_cases],
        "handle_policies": list(HANDLE_POLICIES),
        "baseline_seeds": int(args.baseline_seeds),
        "handle_seeds": int(args.handle_seeds),
        "n_iterations": int(args.n_iterations),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        gate_rows=gate_rows,
        policy_summary=policy_summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-seeds", type=int, default=16)
    parser.add_argument("--handle-seeds", type=int, default=16)
    parser.add_argument("--n-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
