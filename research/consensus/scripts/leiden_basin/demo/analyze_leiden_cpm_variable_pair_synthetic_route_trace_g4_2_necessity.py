#!/usr/bin/env python3
"""Inspect what the strict G4.1 synthetic crossings actually change.

This G4.2 diagnostic consumes the G4.1 route-trace audit and decomposes only
the strict, nonidentical crossing policies. It records source/initial/target
CPM quality, pair state, bridge-context transitions, and sibling-policy
outcomes so that the crossing evidence is not confused with target
reconstruction or a broad wall/pathway claim.

It is a synthetic mechanism audit only. It does not promote basin walls,
compare methods, evaluate quality/cost value, replay NanoClustering, or claim
an algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.clustering.partitioning import partition_class

from analyze_leiden_cpm_variable_pair_synthetic_endpoint_replay import (
    DEFAULT_OUTPUT_DIR as DEFAULT_REPLAY_DIR,
    ENDPOINT_MANIFEST_CSV,
)
from analyze_leiden_cpm_variable_pair_synthetic_route_trace_g4_1_audit import (
    AUDIT_CANDIDATE_SUMMARY_CSV,
    AUDIT_POLICY_SUMMARY_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_G4_1_DIR,
    _coassociation_distance,
    _groups_by_node,
    _initial_membership_for_policy,
    _renumbered_signature,
)
from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    DEFAULT_DESIGN_DIR,
    DESIGN_FAMILY_ROWS_CSV,
    _build_graph,
    _canonical_groups,
    _edges_for_variant,
    _json_safe,
    _mechanism_read,
    _signature_id,
    _synthetic_cases,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_route_trace_g4_2_necessity_v1_20260603"
)

NECESSITY_ROWS_CSV = "variable_pair_synthetic_route_trace_g4_2_necessity_rows.csv"
STAGE_ROWS_CSV = "variable_pair_synthetic_route_trace_g4_2_stage_rows.csv"
BRIDGE_TRANSITIONS_CSV = (
    "variable_pair_synthetic_route_trace_g4_2_bridge_transitions.csv"
)
POLICY_CONTEXT_CSV = "variable_pair_synthetic_route_trace_g4_2_policy_context.csv"
FAMILY_SUMMARY_CSV = "variable_pair_synthetic_route_trace_g4_2_family_summary.csv"
SUMMARY_JSON = "variable_pair_synthetic_route_trace_g4_2_summary.json"
CONFIG_JSON = "variable_pair_synthetic_route_trace_g4_2_config.json"
REPORT_MD = "variable_pair_synthetic_route_trace_g4_2_report.md"

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.2 necessity diagnostic only; strict G4.1 "
    "crossing policies are decomposed into source/initial/target CPM quality, "
    "pair state, bridge transitions, and sibling-policy context. No basin-wall "
    "promotion, no pathway claim, no method comparison, no full NanoClustering "
    "replay, no quality/cost evaluation, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "executed_g4_2_strict_crossing_necessity_audit"
WALL_PROMOTION_STATUS = "not_promoted_strict_crossing_components_only"
METHOD_STATUS = "mechanism_diagnostic_not_method_claim"

HOST_PREFIXES = ("la", "ra")


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


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


def _stage_metrics(
    *,
    case,
    graph,
    membership: list[int],
    stage: str,
) -> dict[str, Any]:
    groups = _canonical_groups(case.nodes, membership)
    read = _mechanism_read(case, membership)
    quality, cluster_count = _quality_for_membership(
        graph=graph,
        gamma=case.gamma,
        membership=membership,
        node_sizes=case.node_sizes,
    )
    labels = {node: int(label) for node, label in zip(case.nodes, membership, strict=True)}
    by_node = _groups_by_node(case.nodes, membership)
    left_group = sorted(by_node["L"])
    right_group = sorted(by_node["R"])
    left_host_count = sum(1 for node in left_group if str(node).startswith("la"))
    right_host_count = sum(1 for node in right_group if str(node).startswith("ra"))
    return {
        "stage": stage,
        "stage_signature_id": _signature_id(groups),
        "stage_signature": json.dumps(groups, sort_keys=True),
        "stage_quality": quality,
        "stage_cluster_count": cluster_count,
        "stage_pair_coassigned": bool(read["pair_coassigned"]),
        "stage_mechanism_read": str(read["mechanism_read"]),
        "stage_left_bridge_same_cluster_count": int(
            read["left_bridge_same_cluster_count"]
        ),
        "stage_right_bridge_same_cluster_count": int(
            read["right_bridge_same_cluster_count"]
        ),
        "stage_left_pair_group": json.dumps(left_group, sort_keys=True),
        "stage_right_pair_group": json.dumps(right_group, sort_keys=True),
        "stage_left_pair_group_size": int(len(left_group)),
        "stage_right_pair_group_size": int(len(right_group)),
        "stage_left_pair_group_host_count": int(left_host_count),
        "stage_right_pair_group_host_count": int(right_host_count),
        "stage_membership": json.dumps(
            {node: int(labels[node]) for node in case.nodes},
            sort_keys=True,
        ),
    }


def _cluster_relation(case, membership: list[int], node: str) -> str:
    group = _groups_by_node(case.nodes, membership)[node]
    has_l = "L" in group
    has_r = "R" in group
    if has_l and has_r:
        return "with_pair"
    if has_l:
        return "with_left_pair_node"
    if has_r:
        return "with_right_pair_node"
    has_left_context = any(str(item).startswith("la") for item in group)
    has_right_context = any(str(item).startswith("ra") for item in group)
    if has_left_context and has_right_context:
        return "with_mixed_context"
    if has_left_context:
        return "with_left_context"
    if has_right_context:
        return "with_right_context"
    if len(group) == 1:
        return "singleton_or_isolated"
    return "with_other_nonpair_nodes"


def _bridge_transition_rows(
    *,
    candidate_row: dict[str, Any],
    case,
    memberships: dict[str, list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_groups = _groups_by_node(case.nodes, memberships["source"])
    initial_groups = _groups_by_node(case.nodes, memberships["initial"])
    target_groups = _groups_by_node(case.nodes, memberships["target"])
    for bridge in case.bridge_nodes:
        bridge = str(bridge)
        source_relation = _cluster_relation(case, memberships["source"], bridge)
        initial_relation = _cluster_relation(case, memberships["initial"], bridge)
        target_relation = _cluster_relation(case, memberships["target"], bridge)
        source_group = sorted(source_groups[bridge])
        initial_group = sorted(initial_groups[bridge])
        target_group = sorted(target_groups[bridge])
        released_to_target_context = (
            source_relation
            in {"with_left_pair_node", "with_right_pair_node", "with_pair"}
            and initial_relation == target_relation
            and target_relation
            in {"with_left_context", "with_right_context", "with_mixed_context"}
        )
        rows.append(
            {
                **candidate_row,
                "bridge_node": bridge,
                "bridge_source_relation": source_relation,
                "bridge_initial_relation": initial_relation,
                "bridge_target_relation": target_relation,
                "bridge_source_group": json.dumps(source_group, sort_keys=True),
                "bridge_initial_group": json.dumps(initial_group, sort_keys=True),
                "bridge_target_group": json.dumps(target_group, sort_keys=True),
                "bridge_changed_source_to_initial": bool(source_group != initial_group),
                "bridge_initial_matches_target": bool(initial_group == target_group),
                "bridge_released_to_target_context": bool(released_to_target_context),
            }
        )
    return rows


def _endpoint_maps(endpoint_manifest: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["endpoint_replay_id"]): row
        for row in endpoint_manifest.to_dict("records")
    }


def _strict_crossing_rows(policy_summary: pd.DataFrame) -> pd.DataFrame:
    rows = policy_summary[
        policy_summary["candidate_type"].eq("pair_relation_change_candidate")
        & policy_summary["g4_1_policy_class"].eq("strict_crosses_to_target")
        & policy_summary["policy_intervention_class"].eq(
            "strict_nonidentical_intervention"
        )
    ].copy()
    return rows.sort_values(
        ["design_family", "route_candidate_id", "trace_policy"],
        kind="stable",
    )


def _policy_context(
    *,
    policy_summary: pd.DataFrame,
    strict_rows: pd.DataFrame,
) -> pd.DataFrame:
    candidate_ids = set(strict_rows["route_candidate_id"].astype(str))
    context = policy_summary[
        policy_summary["route_candidate_id"].astype(str).isin(candidate_ids)
        & policy_summary["candidate_type"].eq("pair_relation_change_candidate")
    ].copy()
    context["is_g4_2_strict_crossing_focus"] = context[
        "g4_1_policy_class"
    ].eq("strict_crosses_to_target")
    context["policy_context_role"] = context["g4_1_policy_class"].map(
        {
            "strict_crosses_to_target": "focus_strict_crossing_policy",
            "strict_bounces_to_source": "source_bounce_sibling_policy",
            "strict_collapses_to_other_known_endpoint": "other_endpoint_sibling_policy",
            "strict_mixed_trace_outcomes": "mixed_sibling_policy",
            "source_identical_noop": "source_noop_control_policy",
            "target_identical_reconstruction": "target_reconstruction_control_policy",
        }
    ).fillna("other_sibling_policy")
    return _claim_columns(
        context.sort_values(
            ["design_family", "route_candidate_id", "trace_policy"],
            kind="stable",
        )
    )


def _policy_rate_lookup(policy_context: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["route_candidate_id"]), str(row["trace_policy"])): row
        for row in policy_context.to_dict("records")
    }


def _rate_for(
    lookup: dict[tuple[str, str], dict[str, Any]],
    route_candidate_id: str,
    trace_policy: str,
    metric: str,
) -> float | None:
    row = lookup.get((route_candidate_id, trace_policy))
    if row is None:
        return None
    return float(row[metric])


def _mechanism_class(row: dict[str, Any]) -> str:
    if (
        row["route_relation"] == "separated_to_coassigned"
        and not row["source_pair_coassigned"]
        and not row["initial_pair_coassigned"]
        and row["target_pair_coassigned"]
        and row["bridge_released_to_target_context_count"] > 0
    ):
        return "bridge_context_release_precedes_pair_merge_synthetic"
    if (
        row["route_relation"] == "separated_to_coassigned"
        and not row["source_pair_coassigned"]
        and row["initial_pair_coassigned"]
        and row["target_pair_coassigned"]
        and row["bridge_released_to_target_context_count"] > 0
    ):
        return "explicit_pair_merge_plus_context_release_synthetic"
    if (
        row["route_relation"] == "separated_to_coassigned"
        and not row["source_pair_coassigned"]
        and row["initial_pair_coassigned"]
        and row["target_pair_coassigned"]
    ):
        return "explicit_pair_merge_without_context_release_synthetic"
    return "strict_crossing_component_unclassified_synthetic"


def _interpret_component_need(row: dict[str, Any]) -> str:
    pair_rate = row.get("pair_relation_only_target_cross_rate")
    bridge_rate = row.get("bridge_context_release_only_target_cross_rate")
    strict_policy = str(row["trace_policy"])
    if strict_policy == "bridge_context_release_only":
        if pair_rate is not None and pair_rate < 0.8:
            return "bridge_context_release_sufficient_in_this_synthetic_trace_pair_only_not_sufficient"
        return "bridge_context_release_sufficient_in_this_synthetic_trace"
    if "context_release" in strict_policy:
        if pair_rate is not None and pair_rate < 0.8 and bridge_rate is not None and bridge_rate < 0.8:
            return "pair_merge_and_context_release_jointly_needed_in_this_synthetic_trace"
        if pair_rate is not None and pair_rate < 0.8:
            return "context_release_adds_to_pair_merge_in_this_synthetic_trace"
    return "component_need_not_resolved_by_g4_2"


def _build_rows(
    *,
    design_dir: Path,
    endpoint_manifest: pd.DataFrame,
    policy_summary: pd.DataFrame,
    strict_rows: pd.DataFrame,
    policy_context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = pd.read_csv(design_dir / DESIGN_FAMILY_ROWS_CSV)
    cases = {case.design_family: case for case in _synthetic_cases(families)}
    endpoints = _endpoint_maps(endpoint_manifest)
    rate_lookup = _policy_rate_lookup(policy_context)
    necessity_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []

    for focus in strict_rows.to_dict("records"):
        family = str(focus["design_family"])
        case = cases[family]
        graph = _build_graph(case.nodes, _edges_for_variant(case, "original"))
        source_endpoint = endpoints[str(focus["source_endpoint_replay_id"])]
        target_endpoint = endpoints[str(focus["target_endpoint_replay_id"])]
        source_signature = str(source_endpoint["endpoint_signature"])
        target_signature = str(target_endpoint["endpoint_signature"])
        source_membership = _initial_membership_for_policy(
            case=case,
            source_signature=source_signature,
            target_signature=target_signature,
            policy="source_replay",
        )
        target_membership = _initial_membership_for_policy(
            case=case,
            source_signature=source_signature,
            target_signature=target_signature,
            policy="target_replay",
        )
        initial_membership = _initial_membership_for_policy(
            case=case,
            source_signature=source_signature,
            target_signature=target_signature,
            policy=str(focus["trace_policy"]),
        )
        memberships = {
            "source": source_membership,
            "initial": initial_membership,
            "target": target_membership,
        }
        stage_by_name = {
            stage: _stage_metrics(
                case=case,
                graph=graph,
                membership=membership,
                stage=stage,
            )
            for stage, membership in memberships.items()
        }
        candidate_base = {
            "route_candidate_id": str(focus["route_candidate_id"]),
            "design_family": family,
            "route_relation": str(focus["route_relation"]),
            "trace_policy": str(focus["trace_policy"]),
            "source_endpoint_replay_id": str(focus["source_endpoint_replay_id"]),
            "target_endpoint_replay_id": str(focus["target_endpoint_replay_id"]),
            "source_endpoint_signature_id": str(
                source_endpoint["endpoint_signature_id"]
            ),
            "target_endpoint_signature_id": str(
                target_endpoint["endpoint_signature_id"]
            ),
            "target_cross_rate": float(focus["target_cross_rate"]),
            "source_bounce_rate": float(focus["source_bounce_rate"]),
            "other_known_collapse_rate": float(focus["other_known_collapse_rate"]),
            "unknown_new_endpoint_rate": float(focus["unknown_new_endpoint_rate"]),
            "g4_1_policy_class": str(focus["g4_1_policy_class"]),
            "policy_initial_signature_id": str(focus["policy_initial_signature_id"]),
            "policy_intervention_class": str(focus["policy_intervention_class"]),
            "changed_nodes_vs_source": int(focus["changed_nodes_vs_source"]),
            "changed_nodes_vs_target": int(focus["changed_nodes_vs_target"]),
            "coassociation_distance_vs_source": int(
                focus["coassociation_distance_vs_source"]
            ),
            "coassociation_distance_vs_target": int(
                focus["coassociation_distance_vs_target"]
            ),
            "source_endpoint_run_share_within_variant": float(
                source_endpoint["endpoint_run_share_within_variant"]
            ),
            "target_endpoint_run_share_within_variant": float(
                target_endpoint["endpoint_run_share_within_variant"]
            ),
        }
        for stage_name, metrics in stage_by_name.items():
            stage_rows.append({**candidate_base, **metrics})

        candidate_bridge_rows = _bridge_transition_rows(
            candidate_row=candidate_base,
            case=case,
            memberships=memberships,
        )
        bridge_rows.extend(candidate_bridge_rows)
        bridge_frame = pd.DataFrame(candidate_bridge_rows)
        bridge_released_count = int(
            bridge_frame["bridge_released_to_target_context"].astype(bool).sum()
        )
        bridge_initial_matches_target_count = int(
            bridge_frame["bridge_initial_matches_target"].astype(bool).sum()
        )
        bridge_changed_count = int(
            bridge_frame["bridge_changed_source_to_initial"].astype(bool).sum()
        )

        source_stage = stage_by_name["source"]
        initial_stage = stage_by_name["initial"]
        target_stage = stage_by_name["target"]
        route_candidate_id = str(focus["route_candidate_id"])
        initial_signature_id = _renumbered_signature(case.nodes, initial_membership)
        target_signature_id = _renumbered_signature(case.nodes, target_membership)
        source_signature_id = _renumbered_signature(case.nodes, source_membership)
        row = {
            **candidate_base,
            "source_signature_id_recomputed": source_signature_id,
            "initial_signature_id_recomputed": initial_signature_id,
            "target_signature_id_recomputed": target_signature_id,
            "source_quality": float(source_stage["stage_quality"]),
            "initial_quality": float(initial_stage["stage_quality"]),
            "target_quality": float(target_stage["stage_quality"]),
            "source_endpoint_quality_median": float(source_endpoint["quality_median"]),
            "target_endpoint_quality_median": float(target_endpoint["quality_median"]),
            "initial_quality_delta_vs_source": float(
                initial_stage["stage_quality"] - source_stage["stage_quality"]
            ),
            "target_quality_delta_vs_source": float(
                target_stage["stage_quality"] - source_stage["stage_quality"]
            ),
            "target_quality_delta_vs_initial": float(
                target_stage["stage_quality"] - initial_stage["stage_quality"]
            ),
            "source_pair_coassigned": bool(source_stage["stage_pair_coassigned"]),
            "initial_pair_coassigned": bool(initial_stage["stage_pair_coassigned"]),
            "target_pair_coassigned": bool(target_stage["stage_pair_coassigned"]),
            "initial_pair_relation_changed_vs_source": bool(
                source_stage["stage_pair_coassigned"]
                != initial_stage["stage_pair_coassigned"]
            ),
            "initial_pair_relation_matches_target": bool(
                initial_stage["stage_pair_coassigned"]
                == target_stage["stage_pair_coassigned"]
            ),
            "source_mechanism_read": str(source_stage["stage_mechanism_read"]),
            "initial_mechanism_read": str(initial_stage["stage_mechanism_read"]),
            "target_mechanism_read": str(target_stage["stage_mechanism_read"]),
            "bridge_count": int(len(case.bridge_nodes)),
            "bridge_changed_source_to_initial_count": bridge_changed_count,
            "bridge_initial_matches_target_count": bridge_initial_matches_target_count,
            "bridge_released_to_target_context_count": bridge_released_count,
            "source_to_initial_coassoc_distance_recomputed": int(
                _coassociation_distance(source_membership, initial_membership)
            ),
            "initial_to_target_coassoc_distance_recomputed": int(
                _coassociation_distance(initial_membership, target_membership)
            ),
            "source_to_target_coassoc_distance": int(
                _coassociation_distance(source_membership, target_membership)
            ),
            "pair_relation_only_target_cross_rate": _rate_for(
                rate_lookup,
                route_candidate_id,
                "pair_relation_only",
                "target_cross_rate",
            ),
            "bridge_context_release_only_target_cross_rate": _rate_for(
                rate_lookup,
                route_candidate_id,
                "bridge_context_release_only",
                "target_cross_rate",
            ),
            "pair_plus_left_context_release_target_cross_rate": _rate_for(
                rate_lookup,
                route_candidate_id,
                "pair_plus_left_context_release",
                "target_cross_rate",
            ),
            "pair_plus_right_context_release_target_cross_rate": _rate_for(
                rate_lookup,
                route_candidate_id,
                "pair_plus_right_context_release",
                "target_cross_rate",
            ),
            "pair_plus_all_context_release_target_cross_rate": _rate_for(
                rate_lookup,
                route_candidate_id,
                "pair_plus_all_context_release",
                "target_cross_rate",
            ),
        }
        row["g4_2_mechanism_class"] = _mechanism_class(row)
        row["g4_2_component_need_read"] = _interpret_component_need(row)
        necessity_rows.append(row)

    return (
        _claim_columns(pd.DataFrame(necessity_rows)),
        _claim_columns(pd.DataFrame(stage_rows)),
        _claim_columns(pd.DataFrame(bridge_rows)),
    )


def _family_summary(necessity_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in necessity_rows.groupby("design_family", sort=True):
        rows.append(
            {
                "design_family": str(family),
                "strict_crossing_focus_count": int(len(group)),
                "route_candidate_count": int(group["route_candidate_id"].nunique()),
                "trace_policy_counts": json.dumps(
                    group["trace_policy"].value_counts().to_dict(),
                    sort_keys=True,
                ),
                "mechanism_class_counts": json.dumps(
                    group["g4_2_mechanism_class"].value_counts().to_dict(),
                    sort_keys=True,
                ),
                "component_need_counts": json.dumps(
                    group["g4_2_component_need_read"].value_counts().to_dict(),
                    sort_keys=True,
                ),
                "initial_pair_changed_count": int(
                    group["initial_pair_relation_changed_vs_source"].astype(bool).sum()
                ),
                "initial_pair_unchanged_count": int(
                    (~group["initial_pair_relation_changed_vs_source"].astype(bool)).sum()
                ),
                "bridge_released_to_target_context_min": int(
                    group["bridge_released_to_target_context_count"].min()
                ),
                "bridge_released_to_target_context_median": float(
                    group["bridge_released_to_target_context_count"].median()
                ),
                "bridge_released_to_target_context_max": int(
                    group["bridge_released_to_target_context_count"].max()
                ),
                "initial_quality_delta_vs_source_min": float(
                    group["initial_quality_delta_vs_source"].min()
                ),
                "initial_quality_delta_vs_source_median": float(
                    group["initial_quality_delta_vs_source"].median()
                ),
                "initial_quality_delta_vs_source_max": float(
                    group["initial_quality_delta_vs_source"].max()
                ),
                "target_quality_delta_vs_source_min": float(
                    group["target_quality_delta_vs_source"].min()
                ),
                "target_quality_delta_vs_source_median": float(
                    group["target_quality_delta_vs_source"].median()
                ),
                "target_quality_delta_vs_source_max": float(
                    group["target_quality_delta_vs_source"].max()
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    g4_1_dir: Path,
    replay_dir: Path,
    output_dir: Path,
    policy_summary: pd.DataFrame,
    strict_rows: pd.DataFrame,
    necessity_rows: pd.DataFrame,
    stage_rows: pd.DataFrame,
    bridge_rows: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "variable_pair_synthetic_route_trace_g4_2_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "g4_1_dir": str(g4_1_dir),
        "replay_dir": str(replay_dir),
        "output_dir": str(output_dir),
        "g4_1_policy_row_count": int(len(policy_summary)),
        "strict_crossing_focus_count": int(len(strict_rows)),
        "necessity_row_count": int(len(necessity_rows)),
        "stage_row_count": int(len(stage_rows)),
        "bridge_transition_row_count": int(len(bridge_rows)),
        "family_count": int(necessity_rows["design_family"].nunique())
        if not necessity_rows.empty
        else 0,
        "trace_policy_counts": necessity_rows["trace_policy"].value_counts().to_dict()
        if not necessity_rows.empty
        else {},
        "mechanism_class_counts": necessity_rows[
            "g4_2_mechanism_class"
        ].value_counts().to_dict()
        if not necessity_rows.empty
        else {},
        "component_need_counts": necessity_rows[
            "g4_2_component_need_read"
        ].value_counts().to_dict()
        if not necessity_rows.empty
        else {},
        "family_rows": family_summary.to_dict("records"),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    necessity_rows: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic Route Trace G4.2 Necessity Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- strict_crossing_focus_count: {summary['strict_crossing_focus_count']}",
        f"- mechanism_class_counts: {summary['mechanism_class_counts']}",
        f"- component_need_counts: {summary['component_need_counts']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Family Summary",
    ]
    for row in family_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.design_family}: focus={row.strict_crossing_focus_count}, "
            f"policies={row.trace_policy_counts}, "
            f"mechanisms={row.mechanism_class_counts}, "
            f"initial_q_delta_median={row.initial_quality_delta_vs_source_median:.6g}, "
            f"target_q_delta_median={row.target_quality_delta_vs_source_median:.6g}"
        )
    lines.extend(["", "## Strict Crossing Components"])
    if necessity_rows.empty:
        lines.append("- none")
    else:
        for row in necessity_rows.itertuples(index=False):
            lines.append(
                "- "
                f"{row.route_candidate_id} {row.design_family} {row.trace_policy}: "
                f"{row.g4_2_mechanism_class}; "
                f"pair source/initial/target="
                f"{row.source_pair_coassigned}/"
                f"{row.initial_pair_coassigned}/"
                f"{row.target_pair_coassigned}; "
                f"bridge_release={row.bridge_released_to_target_context_count}/"
                f"{row.bridge_count}; "
                f"initial_delta={row.initial_quality_delta_vs_source:.6g}; "
                f"target_delta={row.target_quality_delta_vs_source:.6g}; "
                f"read={row.g4_2_component_need_read}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This audit only decomposes the strict G4.1 synthetic crossings. "
                "It does not show a basin wall, a real pathway, a default policy, "
                "or a full-graph method improvement."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    design_dir = Path(args.design_dir)
    g4_1_dir = Path(args.g4_1_dir)
    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint_manifest = pd.read_csv(replay_dir / ENDPOINT_MANIFEST_CSV)
    policy_summary = pd.read_csv(g4_1_dir / AUDIT_POLICY_SUMMARY_CSV)
    candidate_summary = pd.read_csv(g4_1_dir / AUDIT_CANDIDATE_SUMMARY_CSV)
    strict_rows = _strict_crossing_rows(policy_summary)
    if strict_rows.empty:
        raise ValueError("G4.2 requires at least one strict G4.1 crossing row")
    crossing_candidates = set(strict_rows["route_candidate_id"].astype(str))
    missing_candidates = crossing_candidates - set(
        candidate_summary["route_candidate_id"].astype(str)
    )
    if missing_candidates:
        raise ValueError(
            "strict crossing candidates missing from G4.1 candidate summary: "
            f"{sorted(missing_candidates)}"
        )

    policy_context = _policy_context(
        policy_summary=policy_summary,
        strict_rows=strict_rows,
    )
    necessity_rows, stage_rows, bridge_rows = _build_rows(
        design_dir=design_dir,
        endpoint_manifest=endpoint_manifest,
        policy_summary=policy_summary,
        strict_rows=strict_rows,
        policy_context=policy_context,
    )
    family_summary = _family_summary(necessity_rows)

    _write_csv(necessity_rows, output_dir / NECESSITY_ROWS_CSV)
    _write_csv(stage_rows, output_dir / STAGE_ROWS_CSV)
    _write_csv(bridge_rows, output_dir / BRIDGE_TRANSITIONS_CSV)
    _write_csv(policy_context, output_dir / POLICY_CONTEXT_CSV)
    _write_csv(family_summary, output_dir / FAMILY_SUMMARY_CSV)

    summary = _summary(
        g4_1_dir=g4_1_dir,
        replay_dir=replay_dir,
        output_dir=output_dir,
        policy_summary=policy_summary,
        strict_rows=strict_rows,
        necessity_rows=necessity_rows,
        stage_rows=stage_rows,
        bridge_rows=bridge_rows,
        family_summary=family_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_route_trace_g4_2_config.v1",
        "design_dir": str(design_dir),
        "g4_1_dir": str(g4_1_dir),
        "replay_dir": str(replay_dir),
        "output_dir": str(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        necessity_rows=necessity_rows,
        family_summary=family_summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--g4-1-dir", type=Path, default=DEFAULT_G4_1_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
