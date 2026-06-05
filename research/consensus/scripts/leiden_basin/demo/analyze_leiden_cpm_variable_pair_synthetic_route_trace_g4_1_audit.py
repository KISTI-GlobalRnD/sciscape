#!/usr/bin/env python3
"""Audit variable-pair synthetic route traces for target reconstruction leakage.

This G4.1 audit tightens the compact route-trace interpretation. It asks
whether any crossing remains after target-identical initial memberships are
excluded, adds reverse bridge-release policies, records intervention size, and
adds same-pair-state endpoint controls.

It is a trace audit only. It does not promote basin walls, compare methods,
evaluate quality/cost value, replay NanoClustering, or claim an algorithm.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.clustering.runner import LeidenRunner

from analyze_leiden_cpm_variable_pair_synthetic_endpoint_replay import (
    DEFAULT_OUTPUT_DIR as DEFAULT_REPLAY_DIR,
    ENDPOINT_MANIFEST_CSV,
    ROUTE_CANDIDATES_CSV,
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
    _renumber,
    _signature_id,
    _synthetic_cases,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_variable_pair_synthetic_route_trace_g4_1_audit_v1_20260603"
)

AUDIT_RUNS_CSV = "variable_pair_synthetic_route_trace_g4_1_audit_runs.csv"
AUDIT_POLICY_SUMMARY_CSV = "variable_pair_synthetic_route_trace_g4_1_policy_summary.csv"
AUDIT_CANDIDATE_SUMMARY_CSV = "variable_pair_synthetic_route_trace_g4_1_candidate_summary.csv"
AUDIT_CANDIDATES_CSV = "variable_pair_synthetic_route_trace_g4_1_candidates.csv"
SUMMARY_JSON = "variable_pair_synthetic_route_trace_g4_1_summary.json"
CONFIG_JSON = "variable_pair_synthetic_route_trace_g4_1_config.json"
REPORT_MD = "variable_pair_synthetic_route_trace_g4_1_report.md"

AUDIT_POLICIES = (
    "source_replay",
    "target_replay",
    "pair_relation_only",
    "bridge_side_only",
    "bridge_context_release_only",
    "pair_plus_left_bridge_side",
    "pair_plus_right_bridge_side",
    "pair_plus_all_bridge_side",
    "pair_plus_left_context_release",
    "pair_plus_right_context_release",
    "pair_plus_all_context_release",
)
CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.1 route-trace audit only; target-identical "
    "initializations, no-op source initializations, intervention sizes, and "
    "same-state controls are inspected under ordinary Leiden+CPM. No basin-wall "
    "promotion, no method comparison, no full NanoClustering replay, no "
    "quality/cost evaluation, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "executed_g4_1_route_trace_audit"
WALL_PROMOTION_STATUS = "not_promoted_target_reconstruction_audit_only"
METHOD_STATUS = "route_trace_audit_not_method_claim"


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _groups_to_labels(nodes: tuple[str, ...], endpoint_signature: str) -> dict[str, int]:
    groups = json.loads(str(endpoint_signature))
    labels: dict[str, int] = {}
    for label, group in enumerate(groups):
        for node in group:
            labels[str(node)] = int(label)
    unknown = sorted(set(labels) - set(nodes))
    if unknown:
        raise ValueError(f"endpoint signature contains unknown nodes: {unknown}")
    next_label = len(groups)
    for node in nodes:
        if node not in labels:
            labels[node] = next_label
            next_label += 1
    return labels


def _labels_to_membership(nodes: tuple[str, ...], labels: dict[str, int]) -> list[int]:
    return _renumber([int(labels[node]) for node in nodes])


def _groups_by_node(nodes: tuple[str, ...], membership: list[int]) -> dict[str, frozenset[str]]:
    labels: dict[int, set[str]] = {}
    for node, label in zip(nodes, membership, strict=True):
        labels.setdefault(int(label), set()).add(str(node))
    return {
        str(node): frozenset(labels[int(label)])
        for node, label in zip(nodes, membership, strict=True)
    }


def _changed_node_count(
    nodes: tuple[str, ...],
    source_membership: list[int],
    candidate_membership: list[int],
) -> int:
    source_groups = _groups_by_node(nodes, source_membership)
    candidate_groups = _groups_by_node(nodes, candidate_membership)
    return sum(1 for node in nodes if source_groups[node] != candidate_groups[node])


def _coassociation_distance(
    membership_a: list[int],
    membership_b: list[int],
) -> int:
    distance = 0
    for left, right in itertools.combinations(range(len(membership_a)), 2):
        same_a = int(membership_a[left]) == int(membership_a[right])
        same_b = int(membership_b[left]) == int(membership_b[right])
        if same_a != same_b:
            distance += 1
    return distance


def _renumbered_signature(nodes: tuple[str, ...], membership: list[int]) -> str:
    return _signature_id(_canonical_groups(nodes, membership))


def _target_groups_by_node(endpoint_signature: str) -> dict[str, list[str]]:
    groups = json.loads(str(endpoint_signature))
    output: dict[str, list[str]] = {}
    for group in groups:
        group_nodes = [str(node) for node in group]
        for node in group_nodes:
            output[node] = group_nodes
    return output


def _bridge_side(bridge: str) -> str:
    if bridge.startswith("lb") or bridge == "lb":
        return "left"
    if bridge.startswith("rb") or bridge == "rb":
        return "right"
    return "shared"


def _apply_pair_relation(labels: dict[str, int], target_labels: dict[str, int]) -> None:
    if target_labels["L"] == target_labels["R"]:
        labels["R"] = labels["L"]
        return
    if labels["L"] == labels["R"]:
        labels["R"] = max(labels.values()) + 1


def _apply_bridge_side(
    *,
    case,
    labels: dict[str, int],
    target_labels: dict[str, int],
    side: str | None,
) -> None:
    target_left = target_labels["L"]
    target_right = target_labels["R"]
    for bridge in case.bridge_nodes:
        bridge_side = _bridge_side(str(bridge))
        if side is not None and bridge_side != side:
            continue
        if target_labels[bridge] == target_left:
            labels[bridge] = labels["L"]
        elif target_labels[bridge] == target_right:
            labels[bridge] = labels["R"]


def _release_bridge_to_target_context(
    *,
    labels: dict[str, int],
    bridge: str,
    target_groups: dict[str, list[str]],
    target_labels: dict[str, int],
) -> None:
    if target_labels[bridge] in {target_labels["L"], target_labels["R"]}:
        return
    target_group = target_groups[bridge]
    anchors = [node for node in target_group if node != bridge and node in labels]
    if anchors:
        labels[bridge] = labels[anchors[0]]
    else:
        labels[bridge] = max(labels.values()) + 1


def _apply_context_release(
    *,
    case,
    labels: dict[str, int],
    target_labels: dict[str, int],
    target_groups: dict[str, list[str]],
    side: str | None,
) -> None:
    for bridge in case.bridge_nodes:
        bridge_side = _bridge_side(str(bridge))
        if side is not None and bridge_side != side:
            continue
        _release_bridge_to_target_context(
            labels=labels,
            bridge=str(bridge),
            target_groups=target_groups,
            target_labels=target_labels,
        )


def _initial_membership_for_policy(
    *,
    case,
    source_signature: str,
    target_signature: str,
    policy: str,
) -> list[int]:
    source_labels = _groups_to_labels(case.nodes, source_signature)
    target_labels = _groups_to_labels(case.nodes, target_signature)
    target_groups = _target_groups_by_node(target_signature)
    if policy == "source_replay":
        return _labels_to_membership(case.nodes, source_labels)
    if policy == "target_replay":
        return _labels_to_membership(case.nodes, target_labels)

    labels = dict(source_labels)
    if policy == "pair_relation_only":
        _apply_pair_relation(labels, target_labels)
    elif policy == "bridge_side_only":
        _apply_bridge_side(case=case, labels=labels, target_labels=target_labels, side=None)
    elif policy == "bridge_context_release_only":
        _apply_context_release(
            case=case,
            labels=labels,
            target_labels=target_labels,
            target_groups=target_groups,
            side=None,
        )
    elif policy == "pair_plus_left_bridge_side":
        _apply_pair_relation(labels, target_labels)
        _apply_bridge_side(case=case, labels=labels, target_labels=target_labels, side="left")
    elif policy == "pair_plus_right_bridge_side":
        _apply_pair_relation(labels, target_labels)
        _apply_bridge_side(case=case, labels=labels, target_labels=target_labels, side="right")
    elif policy == "pair_plus_all_bridge_side":
        _apply_pair_relation(labels, target_labels)
        _apply_bridge_side(case=case, labels=labels, target_labels=target_labels, side=None)
    elif policy == "pair_plus_left_context_release":
        _apply_pair_relation(labels, target_labels)
        _apply_context_release(
            case=case,
            labels=labels,
            target_labels=target_labels,
            target_groups=target_groups,
            side="left",
        )
    elif policy == "pair_plus_right_context_release":
        _apply_pair_relation(labels, target_labels)
        _apply_context_release(
            case=case,
            labels=labels,
            target_labels=target_labels,
            target_groups=target_groups,
            side="right",
        )
    elif policy == "pair_plus_all_context_release":
        _apply_pair_relation(labels, target_labels)
        _apply_context_release(
            case=case,
            labels=labels,
            target_labels=target_labels,
            target_groups=target_groups,
            side=None,
        )
    else:
        raise ValueError(f"unknown audit policy: {policy}")
    return _labels_to_membership(case.nodes, labels)


def _endpoint_maps(endpoint_manifest: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    by_id = {str(row["endpoint_replay_id"]): row for row in endpoint_manifest.to_dict("records")}
    by_family_sig = {
        (str(row["design_family"]), str(row["endpoint_signature_id"])): row
        for row in endpoint_manifest.to_dict("records")
    }
    return by_id, by_family_sig


def _route_change_candidates(route_candidates: pd.DataFrame) -> pd.DataFrame:
    rows = route_candidates.copy()
    rows["candidate_type"] = "pair_relation_change_candidate"
    rows["candidate_status"] = "g4_1_relation_change_candidate"
    return rows


def _same_state_controls(endpoint_manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    original = endpoint_manifest[endpoint_manifest["graph_variant"].eq("original")]
    for (family, pair_state), group in original.groupby(["design_family", "pair_coassigned"], sort=True):
        records = group.to_dict("records")
        if len(records) < 2:
            continue
        for source in records:
            for target in records:
                if source["endpoint_replay_id"] == target["endpoint_replay_id"]:
                    continue
                rows.append(
                    {
                        "route_candidate_id": f"vp_g4_1_control_{len(rows):04d}",
                        "design_family": str(family),
                        "graph_variant": "original",
                        "route_candidate_status": "g4_1_same_pair_state_control",
                        "route_relation": (
                            "same_coassigned_control"
                            if bool(pair_state)
                            else "same_separated_control"
                        ),
                        "source_endpoint_replay_id": str(source["endpoint_replay_id"]),
                        "target_endpoint_replay_id": str(target["endpoint_replay_id"]),
                        "source_endpoint_signature_id": str(source["endpoint_signature_id"]),
                        "target_endpoint_signature_id": str(target["endpoint_signature_id"]),
                        "source_pair_coassigned": bool(source["pair_coassigned"]),
                        "target_pair_coassigned": bool(target["pair_coassigned"]),
                        "source_mechanism_read": str(source["mechanism_read"]),
                        "target_mechanism_read": str(target["mechanism_read"]),
                        "source_endpoint_run_share_within_variant": float(
                            source["endpoint_run_share_within_variant"]
                        ),
                        "target_endpoint_run_share_within_variant": float(
                            target["endpoint_run_share_within_variant"]
                        ),
                        "source_same_endpoint_replay_rate": 1.0,
                        "target_same_endpoint_replay_rate": 1.0,
                        "recommended_next_probe": "same_pair_state_control",
                        "route_probe_boundary": (
                            "Control only: source and target share the same "
                            "L/R co-assignment state."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _candidate_frame(endpoint_manifest: pd.DataFrame, route_candidates: pd.DataFrame) -> pd.DataFrame:
    route_rows = _route_change_candidates(route_candidates)
    controls = _same_state_controls(endpoint_manifest)
    if controls.empty:
        candidates = route_rows
    else:
        controls["candidate_type"] = "same_pair_state_control"
        controls["candidate_status"] = "g4_1_same_pair_state_control"
        candidates = pd.concat([route_rows, controls], ignore_index=True, sort=False)
    return _claim_columns(candidates)


def _trace_outcome(
    *,
    result_signature_id: str,
    source_signature_id: str,
    target_signature_id: str,
    matched_endpoint: dict[str, Any] | None,
) -> str:
    if result_signature_id == target_signature_id:
        return "crosses_to_target"
    if result_signature_id == source_signature_id:
        return "bounces_to_source"
    if matched_endpoint is not None:
        return "collapses_to_other_known_endpoint"
    return "unknown_new_endpoint"


def _policy_initial_metrics(
    *,
    case,
    source_signature: str,
    target_signature: str,
    policy: str,
) -> dict[str, Any]:
    source = _initial_membership_for_policy(
        case=case,
        source_signature=source_signature,
        target_signature=target_signature,
        policy="source_replay",
    )
    target = _initial_membership_for_policy(
        case=case,
        source_signature=source_signature,
        target_signature=target_signature,
        policy="target_replay",
    )
    initial = _initial_membership_for_policy(
        case=case,
        source_signature=source_signature,
        target_signature=target_signature,
        policy=policy,
    )
    source_sig = _renumbered_signature(case.nodes, source)
    target_sig = _renumbered_signature(case.nodes, target)
    initial_sig = _renumbered_signature(case.nodes, initial)
    pair_count = len(case.nodes) * (len(case.nodes) - 1) // 2
    source_distance = _coassociation_distance(source, initial)
    target_distance = _coassociation_distance(target, initial)
    source_identical = initial_sig == source_sig
    target_identical = initial_sig == target_sig
    if target_identical:
        intervention_class = "target_identical_reconstruction"
    elif source_identical:
        intervention_class = "source_identical_noop"
    else:
        intervention_class = "strict_nonidentical_intervention"
    return {
        "initial_membership": initial,
        "policy_initial_signature_id": initial_sig,
        "policy_source_identical": bool(source_identical),
        "policy_target_identical": bool(target_identical),
        "policy_intervention_class": intervention_class,
        "changed_nodes_vs_source": int(_changed_node_count(case.nodes, source, initial)),
        "changed_nodes_vs_target": int(_changed_node_count(case.nodes, target, initial)),
        "coassociation_distance_vs_source": int(source_distance),
        "coassociation_distance_vs_target": int(target_distance),
        "coassociation_distance_vs_source_share": (
            float(source_distance / pair_count) if pair_count else 0.0
        ),
        "coassociation_distance_vs_target_share": (
            float(target_distance / pair_count) if pair_count else 0.0
        ),
    }


def _run_audit(
    *,
    design_dir: Path,
    endpoint_manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    trace_seeds: int,
    n_iterations: int,
) -> pd.DataFrame:
    families = pd.read_csv(design_dir / DESIGN_FAMILY_ROWS_CSV)
    cases = {case.design_family: case for case in _synthetic_cases(families)}
    by_id, by_family_sig = _endpoint_maps(endpoint_manifest)
    rows: list[dict[str, Any]] = []

    for candidate in candidates.itertuples(index=False):
        family = str(candidate.design_family)
        case = cases[family]
        graph = _build_graph(case.nodes, _edges_for_variant(case, "original"))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        source = by_id[str(candidate.source_endpoint_replay_id)]
        target = by_id[str(candidate.target_endpoint_replay_id)]
        source_signature = str(source["endpoint_signature"])
        target_signature = str(target["endpoint_signature"])
        source_signature_id = str(source["endpoint_signature_id"])
        target_signature_id = str(target["endpoint_signature_id"])
        for policy in AUDIT_POLICIES:
            metrics = _policy_initial_metrics(
                case=case,
                source_signature=source_signature,
                target_signature=target_signature,
                policy=policy,
            )
            initial = metrics.pop("initial_membership")
            for seed in range(int(trace_seeds)):
                result = runner.run(
                    case.gamma,
                    seed=int(seed),
                    initial_membership=initial,
                    node_sizes=case.node_sizes,
                )
                membership = list(map(int, result.membership))
                groups = _canonical_groups(case.nodes, membership)
                result_signature_id = _signature_id(groups)
                matched = by_family_sig.get((family, result_signature_id))
                read = _mechanism_read(case, membership)
                rows.append(
                    {
                        "route_candidate_id": str(candidate.route_candidate_id),
                        "candidate_type": str(candidate.candidate_type),
                        "design_family": family,
                        "route_relation": str(candidate.route_relation),
                        "trace_policy": policy,
                        "trace_seed": int(seed),
                        "source_endpoint_replay_id": str(candidate.source_endpoint_replay_id),
                        "target_endpoint_replay_id": str(candidate.target_endpoint_replay_id),
                        "source_endpoint_signature_id": source_signature_id,
                        "target_endpoint_signature_id": target_signature_id,
                        "result_endpoint_signature_id": result_signature_id,
                        "result_endpoint_replay_id": (
                            None if matched is None else str(matched["endpoint_replay_id"])
                        ),
                        "trace_outcome": _trace_outcome(
                            result_signature_id=result_signature_id,
                            source_signature_id=source_signature_id,
                            target_signature_id=target_signature_id,
                            matched_endpoint=matched,
                        ),
                        "result_pair_coassigned": bool(read["pair_coassigned"]),
                        "result_mechanism_read": str(read["mechanism_read"]),
                        "result_quality": float(result.quality),
                        "result_cluster_count": int(result.cluster_count),
                        "result_endpoint_signature": json.dumps(groups, sort_keys=True),
                        **metrics,
                    }
                )
    return _claim_columns(pd.DataFrame(rows))


def _classify_policy(group: pd.DataFrame) -> str:
    target_rate = float(group["trace_outcome"].eq("crosses_to_target").mean())
    source_rate = float(group["trace_outcome"].eq("bounces_to_source").mean())
    other_rate = float(group["trace_outcome"].eq("collapses_to_other_known_endpoint").mean())
    unknown_rate = float(group["trace_outcome"].eq("unknown_new_endpoint").mean())
    target_identical = bool(group["policy_target_identical"].iloc[0])
    source_identical = bool(group["policy_source_identical"].iloc[0])
    if target_identical:
        return "target_identical_reconstruction"
    if source_identical:
        return "source_identical_noop"
    if target_rate >= 0.8:
        return "strict_crosses_to_target"
    if source_rate >= 0.8:
        return "strict_bounces_to_source"
    if other_rate >= 0.8:
        return "strict_collapses_to_other_known_endpoint"
    if unknown_rate >= 0.8:
        return "strict_unknown_new_endpoint"
    return "strict_mixed_trace_outcomes"


def _policy_summary(audit_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_candidate_id",
        "candidate_type",
        "design_family",
        "route_relation",
        "trace_policy",
        "source_endpoint_replay_id",
        "target_endpoint_replay_id",
        "policy_initial_signature_id",
        "policy_intervention_class",
        "policy_source_identical",
        "policy_target_identical",
        "changed_nodes_vs_source",
        "changed_nodes_vs_target",
        "coassociation_distance_vs_source",
        "coassociation_distance_vs_target",
        "coassociation_distance_vs_source_share",
        "coassociation_distance_vs_target_share",
    ]
    for keys, group in audit_runs.groupby(group_cols, sort=True):
        key = dict(zip(group_cols, keys, strict=True))
        rows.append(
            {
                **key,
                "run_count": int(len(group)),
                "target_cross_count": int(group["trace_outcome"].eq("crosses_to_target").sum()),
                "source_bounce_count": int(group["trace_outcome"].eq("bounces_to_source").sum()),
                "other_known_collapse_count": int(
                    group["trace_outcome"].eq("collapses_to_other_known_endpoint").sum()
                ),
                "unknown_new_endpoint_count": int(
                    group["trace_outcome"].eq("unknown_new_endpoint").sum()
                ),
                "target_cross_rate": float(group["trace_outcome"].eq("crosses_to_target").mean()),
                "source_bounce_rate": float(group["trace_outcome"].eq("bounces_to_source").mean()),
                "other_known_collapse_rate": float(
                    group["trace_outcome"].eq("collapses_to_other_known_endpoint").mean()
                ),
                "unknown_new_endpoint_rate": float(
                    group["trace_outcome"].eq("unknown_new_endpoint").mean()
                ),
                "distinct_result_endpoint_count": int(group["result_endpoint_signature_id"].nunique()),
                "g4_1_policy_class": _classify_policy(group),
                "result_quality_min": float(group["result_quality"].min()),
                "result_quality_median": float(group["result_quality"].median()),
                "result_quality_max": float(group["result_quality"].max()),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _candidate_summary(policy_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_candidate_id",
        "candidate_type",
        "design_family",
        "route_relation",
        "source_endpoint_replay_id",
        "target_endpoint_replay_id",
    ]
    for keys, full_group in policy_summary.groupby(group_cols, sort=True):
        key = dict(zip(group_cols, keys, strict=True))
        group = full_group[
            full_group["policy_intervention_class"].eq("strict_nonidentical_intervention")
        ]
        has_strict = not group.empty
        rows.append(
            {
                **key,
                "strict_policy_count": int(len(group)),
                "strict_crossing_policy_count": int(
                    group["g4_1_policy_class"].eq("strict_crosses_to_target").sum()
                    if has_strict
                    else 0
                ),
                "strict_bounce_policy_count": int(
                    group["g4_1_policy_class"].eq("strict_bounces_to_source").sum()
                    if has_strict
                    else 0
                ),
                "strict_mixed_policy_count": int(
                    group["g4_1_policy_class"].eq("strict_mixed_trace_outcomes").sum()
                    if has_strict
                    else 0
                ),
                "best_strict_target_cross_rate": (
                    float(group["target_cross_rate"].max()) if has_strict else 0.0
                ),
                "best_strict_source_bounce_rate": (
                    float(group["source_bounce_rate"].max()) if has_strict else 0.0
                ),
                "min_strict_coassociation_distance_vs_source": (
                    int(group["coassociation_distance_vs_source"].min()) if has_strict else 0
                ),
                "min_strict_coassociation_distance_vs_target": (
                    int(group["coassociation_distance_vs_target"].min()) if has_strict else 0
                ),
                "g4_1_candidate_status": (
                    "has_strict_nonidentical_crossing_policy"
                    if has_strict
                    and group["g4_1_policy_class"].eq("strict_crosses_to_target").any()
                    else "no_strict_nonidentical_crossing_policy"
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    replay_dir: Path,
    output_dir: Path,
    candidates: pd.DataFrame,
    audit_runs: pd.DataFrame,
    policy_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
) -> dict[str, Any]:
    relation_candidates = candidate_summary[
        candidate_summary["candidate_type"].eq("pair_relation_change_candidate")
    ]
    controls = candidate_summary[candidate_summary["candidate_type"].eq("same_pair_state_control")]
    return {
        "schema": "variable_pair_synthetic_route_trace_g4_1_summary.v1",
        "status": "executed_g4_1_route_trace_audit",
        "replay_dir": str(replay_dir),
        "output_dir": str(output_dir),
        "candidate_count": int(len(candidates)),
        "relation_candidate_count": int(
            candidates["candidate_type"].eq("pair_relation_change_candidate").sum()
        ),
        "same_state_control_count": int(
            candidates["candidate_type"].eq("same_pair_state_control").sum()
        ),
        "audit_run_count": int(len(audit_runs)),
        "policy_class_counts": policy_summary["g4_1_policy_class"].value_counts().to_dict(),
        "relation_candidate_status_counts": relation_candidates[
            "g4_1_candidate_status"
        ].value_counts().to_dict(),
        "same_state_control_status_counts": controls[
            "g4_1_candidate_status"
        ].value_counts().to_dict(),
        "target_identical_policy_count": int(policy_summary["policy_target_identical"].sum()),
        "source_identical_policy_count": int(policy_summary["policy_source_identical"].sum()),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    candidate_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
) -> None:
    relation_candidates = candidate_summary[
        candidate_summary["candidate_type"].eq("pair_relation_change_candidate")
    ]
    lines = [
        "# Variable-Pair Synthetic Route Trace G4.1 Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- candidate_count: {summary['candidate_count']}",
        f"- relation_candidate_count: {summary['relation_candidate_count']}",
        f"- same_state_control_count: {summary['same_state_control_count']}",
        f"- audit_run_count: {summary['audit_run_count']}",
        f"- policy_class_counts: {summary['policy_class_counts']}",
        f"- relation_candidate_status_counts: {summary['relation_candidate_status_counts']}",
        f"- same_state_control_status_counts: {summary['same_state_control_status_counts']}",
        f"- target_identical_policy_count: {summary['target_identical_policy_count']}",
        f"- source_identical_policy_count: {summary['source_identical_policy_count']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Relation Candidates",
    ]
    for row in relation_candidates.itertuples(index=False):
        lines.append(
            "- "
            f"{row.route_candidate_id} {row.design_family} {row.route_relation}: "
            f"{row.g4_1_candidate_status}, "
            f"best_strict_cross={row.best_strict_target_cross_rate:.3f}, "
            f"best_strict_bounce={row.best_strict_source_bounce_rate:.3f}"
        )
    lines.extend(["", "## Strict Crossing Policies"])
    strict_cross = policy_summary[
        policy_summary["g4_1_policy_class"].eq("strict_crosses_to_target")
    ]
    if strict_cross.empty:
        lines.append("- none")
    else:
        for row in strict_cross.itertuples(index=False):
            lines.append(
                "- "
                f"{row.route_candidate_id} {row.trace_policy}: "
                f"target={row.target_cross_rate:.3f}, "
                f"source_distance={row.coassociation_distance_vs_source}, "
                f"target_distance={row.coassociation_distance_vs_target}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This audit separates target reconstruction from strict "
                "nonidentical route traces. Strict crossing, if any, remains a "
                "synthetic trace diagnostic only."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    design_dir = Path(args.design_dir)
    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_manifest = pd.read_csv(replay_dir / ENDPOINT_MANIFEST_CSV)
    route_candidates = pd.read_csv(replay_dir / ROUTE_CANDIDATES_CSV)
    candidates = _candidate_frame(endpoint_manifest, route_candidates)
    if not bool(args.include_same_state_controls):
        candidates = candidates[
            candidates["candidate_type"].eq("pair_relation_change_candidate")
        ].copy()
    audit_runs = _run_audit(
        design_dir=design_dir,
        endpoint_manifest=endpoint_manifest,
        candidates=candidates,
        trace_seeds=int(args.trace_seeds),
        n_iterations=int(args.n_iterations),
    )
    policy_summary = _policy_summary(audit_runs)
    candidate_summary = _candidate_summary(policy_summary)
    _write_csv(candidates, output_dir / AUDIT_CANDIDATES_CSV)
    _write_csv(audit_runs, output_dir / AUDIT_RUNS_CSV)
    _write_csv(policy_summary, output_dir / AUDIT_POLICY_SUMMARY_CSV)
    _write_csv(candidate_summary, output_dir / AUDIT_CANDIDATE_SUMMARY_CSV)
    summary = _summary(
        replay_dir=replay_dir,
        output_dir=output_dir,
        candidates=candidates,
        audit_runs=audit_runs,
        policy_summary=policy_summary,
        candidate_summary=candidate_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_route_trace_g4_1_config.v1",
        "design_dir": str(design_dir),
        "replay_dir": str(replay_dir),
        "output_dir": str(output_dir),
        "audit_policies": list(AUDIT_POLICIES),
        "trace_seeds": int(args.trace_seeds),
        "n_iterations": int(args.n_iterations),
        "include_same_state_controls": bool(args.include_same_state_controls),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        candidate_summary=candidate_summary,
        policy_summary=policy_summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trace-seeds", type=int, default=16)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--include-same-state-controls", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
