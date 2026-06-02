#!/usr/bin/env python3
"""Run Stress 4 mechanism-variant panel P5-P8 from phase-locked P0-P4 inputs.

This runner is intentionally downstream of the P0-P4 candidate registry and
P4.5 control audit. It verifies the phase lock before reading endpoints, then
runs ordinary Leiden + CPM seed sweeps, freezes recurrent endpoint signatures,
executes the phase-locked blind candidates, and compares first hits against a
random-restart discovery baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import igraph as ig
import pandas as pd

from sciscape.clustering.runner import LeidenRunner

from run_leiden_cpm_tiny_demo_seed_sweep import (
    _canonical_groups,
    _json_safe,
    _signature_id,
    _write_csv,
)


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_INPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_v1_1_20260601"
DEFAULT_AUDIT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_p4_5_control_audit_v1_1_20260601"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_p5_p8_v1_1_20260601"

GRAPH_MANIFEST_CSV = "tiny_cpm_variant_graph_manifest.csv"
GRAPH_EDGES_CSV = "tiny_cpm_variant_graph_edges.csv"
GRAPH_ROLES_CSV = "tiny_cpm_variant_graph_roles.csv"
BLIND_CANDIDATE_REGISTRY_CSV = "tiny_cpm_variant_blind_candidate_registry.csv"
PHASE_LOCK_JSON = "tiny_cpm_variant_phase_lock.json"
P4_5_SUMMARY_JSON = "tiny_cpm_variant_p4_5_summary.json"

SEED_RUNS_CSV = "tiny_cpm_variant_p5_seed_runs.csv"
ENDPOINT_SUMMARY_CSV = "tiny_cpm_variant_p5_endpoint_summary.csv"
FROZEN_ENDPOINT_MANIFEST_CSV = "tiny_cpm_variant_p6_frozen_endpoint_manifest.csv"
BASELINE_DISCOVERY_CSV = "tiny_cpm_variant_p6_baseline_endpoint_discovery.csv"
CANDIDATE_ATTEMPTS_CSV = "tiny_cpm_variant_p7_candidate_attempts.csv"
ENDPOINT_FIRST_HITS_CSV = "tiny_cpm_variant_p7_endpoint_first_hits.csv"
TARGET_ATTRIBUTION_CSV = "tiny_cpm_variant_p7_target_attribution.csv"
GATE_MATRIX_CSV = "tiny_cpm_variant_p8_gate_matrix.csv"
SUMMARY_JSON = "tiny_cpm_variant_p5_p8_summary.json"
CONFIG_JSON = "tiny_cpm_variant_p5_p8_config.json"
REPORT_MD = "tiny_cpm_variant_p5_p8_report.md"

CLAIM_BOUNDARY = (
    "Tiny CPM mechanism-variant P5-P8 endpoint diagnostic only; reads "
    "phase-locked P0-P4 graph/candidate artifacts after P4.5 control audit, "
    "runs ordinary Leiden + CPM endpoint evaluation and candidate initialization "
    "attempts, no route/pathway execution, no wall promotion, no quality/cost "
    "claim, no NanoClustering generality claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_p5_p8_endpoint_diagnostic_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "candidate_initialization_endpoint_diagnostic_not_algorithm_claim"

DEFAULT_BUDGETS = (1, 2, 3, 5, 10, 20)


@dataclass(frozen=True)
class VariantInput:
    variant_id: str
    mechanism_family: str
    mechanism_state: str
    gamma: float
    seed_count: int
    expected_baseline_behavior: str
    responsible_rule: str
    graph: ig.Graph
    roles: pd.DataFrame


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_phase_lock(input_dir: Path) -> tuple[dict[str, Any], bool, list[str], dict[str, str]]:
    phase_lock = _load_json(input_dir / PHASE_LOCK_JSON)
    mismatches: list[str] = []
    observed: dict[str, str] = {}
    for filename, expected_hash in sorted(phase_lock.get("artifact_hashes", {}).items()):
        path = input_dir / filename
        if not path.exists():
            mismatches.append(f"{filename}:missing")
            continue
        digest = _sha256_file(path)
        observed[filename] = digest
        if digest != expected_hash:
            mismatches.append(f"{filename}:hash_mismatch")
    return phase_lock, not mismatches, mismatches, observed


def _load_inputs(input_dir: Path, audit_dir: Path) -> dict[str, Any]:
    required = [
        GRAPH_MANIFEST_CSV,
        GRAPH_EDGES_CSV,
        GRAPH_ROLES_CSV,
        BLIND_CANDIDATE_REGISTRY_CSV,
        PHASE_LOCK_JSON,
    ]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing P0-P4 input artifacts in {_rel(input_dir)}: {missing}")
    audit_summary_path = audit_dir / P4_5_SUMMARY_JSON
    if not audit_summary_path.exists():
        raise FileNotFoundError(f"missing P4.5 audit summary: {_rel(audit_summary_path)}")
    phase_lock, phase_lock_verified, phase_lock_mismatches, observed_hashes = _verify_phase_lock(input_dir)
    return {
        "manifest": pd.read_csv(input_dir / GRAPH_MANIFEST_CSV),
        "edges": pd.read_csv(input_dir / GRAPH_EDGES_CSV),
        "roles": pd.read_csv(input_dir / GRAPH_ROLES_CSV),
        "registry": pd.read_csv(input_dir / BLIND_CANDIDATE_REGISTRY_CSV),
        "phase_lock": phase_lock,
        "phase_lock_verified": phase_lock_verified,
        "phase_lock_mismatches": phase_lock_mismatches,
        "observed_hashes": observed_hashes,
        "audit_summary": _load_json(audit_summary_path),
    }


def _graph_from_edges(edges: pd.DataFrame, roles: pd.DataFrame) -> ig.Graph:
    role_nodes = {
        node
        for value in roles["node_ids"].astype(str)
        for node in json.loads(value)
    }
    edge_nodes = set(edges["left_node"].astype(str)) | set(edges["right_node"].astype(str))
    names = sorted(role_nodes | edge_nodes)
    index = {name: offset for offset, name in enumerate(names)}
    graph = ig.Graph(
        n=len(names),
        edges=[(index[str(row.left_node)], index[str(row.right_node)]) for row in edges.itertuples(index=False)],
        directed=False,
    )
    graph.vs["name"] = names
    graph.es["weight"] = [float(row.weight) for row in edges.itertuples(index=False)]
    return graph


def _variant_inputs(manifest: pd.DataFrame, edges: pd.DataFrame, roles: pd.DataFrame) -> list[VariantInput]:
    variants: list[VariantInput] = []
    for row in manifest.sort_values("variant_id").itertuples(index=False):
        variant_id = str(row.variant_id)
        variant_edges = edges[edges["variant_id"].astype(str).eq(variant_id)].copy()
        variant_roles = roles[roles["variant_id"].astype(str).eq(variant_id)].copy()
        variants.append(
            VariantInput(
                variant_id=variant_id,
                mechanism_family=str(row.mechanism_family),
                mechanism_state=str(row.mechanism_state),
                gamma=float(row.gamma),
                seed_count=int(row.seed_count),
                expected_baseline_behavior=str(row.expected_baseline_behavior),
                responsible_rule=str(row.responsible_rule),
                graph=_graph_from_edges(variant_edges, variant_roles),
                roles=variant_roles,
            )
        )
    return variants


def _role_nodes(roles: pd.DataFrame, role_type: str, role_slot: str | None = None) -> tuple[str, ...]:
    rows = roles[roles["role_type"].astype(str).eq(role_type)]
    if role_slot is not None:
        rows = rows[rows["role_slot"].astype(str).eq(role_slot)]
    nodes: list[str] = []
    for value in rows["node_ids"].astype(str):
        nodes.extend(str(node) for node in json.loads(value))
    return tuple(sorted(dict.fromkeys(nodes)))


def _role_slots(roles: pd.DataFrame, role_type: str) -> list[str]:
    rows = roles[roles["role_type"].astype(str).eq(role_type)]
    return sorted(rows["role_slot"].astype(str).unique().tolist())


def _membership_by_node(graph: ig.Graph, membership: list[int]) -> dict[str, int]:
    return {str(name): int(label) for name, label in zip(graph.vs["name"], membership)}


def _cluster_nodes(graph: ig.Graph, membership: list[int], node: str) -> tuple[str, ...]:
    labels = _membership_by_node(graph, membership)
    target_label = labels[node]
    return tuple(sorted(name for name, label in labels.items() if label == target_label))


def _host_slots_in_nodes(nodes: tuple[str, ...], roles: pd.DataFrame) -> tuple[str, ...]:
    present: list[str] = []
    node_set = set(nodes)
    for slot in _role_slots(roles, "host"):
        if node_set & set(_role_nodes(roles, "host", slot)):
            present.append(slot)
    return tuple(sorted(present))


def _variant_mechanism_read(variant: VariantInput, membership: list[int]) -> str:
    roles = variant.roles
    graph = variant.graph
    labels = _membership_by_node(graph, membership)

    if variant.mechanism_family == "near_tie_bridge":
        bridge_nodes = _role_nodes(roles, "bridge", "bridge")
        if not bridge_nodes:
            return "bridge_missing"
        peers = _cluster_nodes(graph, membership, bridge_nodes[0])
        hosts = _host_slots_in_nodes(peers, roles)
        if len(hosts) == 1:
            return f"bridge_to_{hosts[0]}"
        if len(hosts) == 0:
            return "bridge_separate"
        return "bridge_between_multiple_hosts"

    if variant.mechanism_family == "absorption_triad":
        module_nodes = _role_nodes(roles, "small_module", "small_module")
        module_labels = {labels[node] for node in module_nodes}
        if len(module_labels) > 1:
            return "small_module_split"
        peers = _cluster_nodes(graph, membership, module_nodes[0])
        hosts = _host_slots_in_nodes(peers, roles)
        if len(hosts) == 1:
            return f"small_module_absorbed_by_{hosts[0]}"
        if len(hosts) == 0:
            return "small_module_separate"
        return "small_module_between_multiple_hosts"

    if variant.mechanism_family == "balanced_split":
        middle_nodes = _role_nodes(roles, "middle_module", "middle_module")
        middle_labels = {labels[node] for node in middle_nodes}
        pull_a = _role_nodes(roles, "middle_submodule", "middle_pull_a")
        pull_b = _role_nodes(roles, "middle_submodule", "middle_pull_b")
        if pull_a and pull_b:
            pull_a_hosts = set()
            pull_b_hosts = set()
            for node in pull_a:
                pull_a_hosts.update(_host_slots_in_nodes(_cluster_nodes(graph, membership, node), roles))
            for node in pull_b:
                pull_b_hosts.update(_host_slots_in_nodes(_cluster_nodes(graph, membership, node), roles))
            if pull_a_hosts and pull_b_hosts and pull_a_hosts.isdisjoint(pull_b_hosts):
                return "middle_contact_split"
        if len(middle_labels) == 1:
            peers = _cluster_nodes(graph, membership, middle_nodes[0])
            hosts = _host_slots_in_nodes(peers, roles)
            if len(hosts) == 0:
                return "middle_module_separate"
            if len(hosts) == 1:
                return f"middle_module_absorbed_by_{hosts[0]}"
            return "middle_module_between_multiple_hosts"
        return "mixed_middle_fragmentation"

    if variant.mechanism_family == "diffuse_fragment":
        weak_nodes = _role_nodes(roles, "weak_module", "weak_module")
        weak_labels = {labels[node] for node in weak_nodes}
        host_targets: set[str] = set()
        for node in weak_nodes:
            host_targets.update(_host_slots_in_nodes(_cluster_nodes(graph, membership, node), roles))
        if len(weak_labels) == 1:
            if host_targets:
                return "weak_module_absorbed_or_merged"
            return "weak_module_separate"
        if len(host_targets) >= 2:
            return "weak_pair_tail_split_or_fragmented"
        return "mixed_weak_module_fragmentation"

    return "unclassified"


def _initial_membership(node_names: list[str], groups: tuple[tuple[str, ...], ...]) -> list[int]:
    assigned: dict[str, int] = {}
    for label, group in enumerate(groups):
        for node in group:
            assigned[str(node)] = int(label)
    next_label = len(groups)
    for node in node_names:
        if node not in assigned:
            assigned[node] = next_label
            next_label += 1
    return [assigned[node] for node in node_names]


def _target_class_compatible(target_mechanism_class: str, mechanism_read: str | None) -> bool:
    if mechanism_read is None:
        return False
    target = str(target_mechanism_class)
    read = str(mechanism_read)
    if target.startswith("bridge_to_"):
        return read == target
    if target.startswith("small_module_to_host_a_boundary_core"):
        return read == "small_module_absorbed_by_host_a"
    if target.startswith("small_module_to_host_b_boundary_core"):
        return read == "small_module_absorbed_by_host_b"
    if target == "middle_contact_split":
        return read == "middle_contact_split"
    if target.startswith("middle_module_to_host_a_boundary_core"):
        return read == "middle_module_absorbed_by_host_a"
    if target.startswith("middle_module_to_host_b_boundary_core"):
        return read == "middle_module_absorbed_by_host_b"
    if target == "weak_pair_tail_split":
        return read == "weak_pair_tail_split_or_fragmented"
    if target == "joint_weak_pair_tail_split":
        return read == "weak_pair_tail_split_or_fragmented"
    return False


def _run_baseline(
    variants: list[VariantInput],
    *,
    seeds: int,
    n_iterations: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        runner = LeidenRunner(variant.graph, objective="cpm", default_iterations=n_iterations)
        for seed in range(seeds):
            result = runner.run(variant.gamma, seed=seed)
            membership = list(map(int, result.membership))
            groups = _canonical_groups(variant.graph, membership)
            rows.append(
                {
                    "variant_id": variant.variant_id,
                    "mechanism_family": variant.mechanism_family,
                    "mechanism_state": variant.mechanism_state,
                    "responsible_rule": variant.responsible_rule,
                    "expected_baseline_behavior": variant.expected_baseline_behavior,
                    "gamma": variant.gamma,
                    "seed": int(seed),
                    "node_count": int(variant.graph.vcount()),
                    "edge_count": int(variant.graph.ecount()),
                    "cluster_count": int(result.cluster_count),
                    "quality": float(result.quality),
                    "endpoint_signature_id": _signature_id(groups),
                    "endpoint_signature": json.dumps(groups, sort_keys=True),
                    "mechanism_read": _variant_mechanism_read(variant, membership),
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "seed"]))


def _endpoint_summary(seed_runs: pd.DataFrame, *, seeds: int, recurrent_threshold: int) -> pd.DataFrame:
    rows = (
        seed_runs.groupby(
            [
                "variant_id",
                "mechanism_family",
                "mechanism_state",
                "responsible_rule",
                "expected_baseline_behavior",
                "gamma",
                "endpoint_signature_id",
                "endpoint_signature",
                "mechanism_read",
            ],
            as_index=False,
        )
        .agg(
            seed_count=("seed", "nunique"),
            example_seed=("seed", "min"),
            node_count=("node_count", "min"),
            edge_count=("edge_count", "min"),
            quality_min=("quality", "min"),
            quality_median=("quality", "median"),
            quality_max=("quality", "max"),
            cluster_count_min=("cluster_count", "min"),
            cluster_count_max=("cluster_count", "max"),
        )
    )
    rows["seed_share"] = rows["seed_count"] / seeds
    rows["is_recurrent_endpoint"] = rows["seed_count"].ge(recurrent_threshold)
    rows = rows.sort_values(
        ["variant_id", "seed_count", "quality_median", "endpoint_signature_id"],
        ascending=[True, False, False, True],
    )
    rows["endpoint_rank_in_variant"] = rows.groupby("variant_id").cumcount() + 1
    return _with_claim_columns(rows)


def _frozen_endpoint_manifest(endpoint_summary: pd.DataFrame, *, panel_version: str) -> pd.DataFrame:
    rows = endpoint_summary.copy()
    rows["frozen_endpoint_id"] = [
        f"{variant}__endpoint{int(rank):02d}"
        for variant, rank in zip(rows["variant_id"], rows["endpoint_rank_in_variant"])
    ]
    rows["freeze_status"] = f"frozen_stress4_variant_endpoint_universe_{panel_version}"
    rows["freeze_scope"] = f"plain_leiden_cpm_mechanism_variant_panel_{panel_version}"
    rows["baseline_role"] = rows["is_recurrent_endpoint"].map(
        {True: "recurrent_baseline_endpoint", False: "rare_baseline_endpoint"}
    )
    columns = [
        "frozen_endpoint_id",
        "freeze_status",
        "freeze_scope",
        "baseline_role",
        "variant_id",
        "mechanism_family",
        "mechanism_state",
        "responsible_rule",
        "gamma",
        "endpoint_rank_in_variant",
        "endpoint_signature_id",
        "mechanism_read",
        "seed_count",
        "seed_share",
        "is_recurrent_endpoint",
        "quality_min",
        "quality_median",
        "quality_max",
        "cluster_count_min",
        "cluster_count_max",
        "endpoint_signature",
    ]
    return _with_claim_columns(rows[columns])


def _baseline_endpoint_discovery(
    seed_runs: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    permutations: int,
    rng_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    recurrent = manifest[manifest["is_recurrent_endpoint"].astype(bool)].copy()
    for endpoint in recurrent.itertuples(index=False):
        variant_id = str(endpoint.variant_id)
        endpoint_id = str(endpoint.endpoint_signature_id)
        variant_runs = seed_runs[seed_runs["variant_id"].astype(str).eq(variant_id)]
        seed_to_endpoint = {
            int(row.seed): str(row.endpoint_signature_id)
            for row in variant_runs[["seed", "endpoint_signature_id"]].itertuples(index=False)
        }
        seed_values = sorted(seed_to_endpoint)
        first_positions: list[int] = []
        for permutation in range(permutations):
            rng = random.Random(f"{rng_seed}:{variant_id}:{endpoint_id}:{permutation}:{len(seed_values)}")
            order = seed_values[:]
            rng.shuffle(order)
            first = next(
                (index + 1 for index, seed in enumerate(order) if seed_to_endpoint[seed] == endpoint_id),
                len(order) + 1,
            )
            first_positions.append(first)
        first_positions_sorted = sorted(first_positions)
        def quantile(q: float) -> float:
            if not first_positions_sorted:
                return math.nan
            offset = min(len(first_positions_sorted) - 1, max(0, math.ceil(q * len(first_positions_sorted)) - 1))
            return float(first_positions_sorted[offset])

        rows.append(
            {
                "variant_id": variant_id,
                "frozen_endpoint_id": str(endpoint.frozen_endpoint_id),
                "endpoint_signature_id": endpoint_id,
                "mechanism_read": str(endpoint.mechanism_read),
                "seed_count": int(endpoint.seed_count),
                "seed_share": float(endpoint.seed_share),
                "permutations": int(permutations),
                "baseline_first_hit_p50": quantile(0.50),
                "baseline_first_hit_p75": quantile(0.75),
                "baseline_first_hit_p90": quantile(0.90),
                "baseline_first_hit_max": float(max(first_positions_sorted)),
                "rng_seed": int(rng_seed),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "frozen_endpoint_id"]))


def _manifest_by_signature(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(variant): group.set_index("endpoint_signature_id", drop=False)
        for variant, group in manifest.groupby("variant_id", sort=True)
    }


def _candidate_attempts(
    variants: list[VariantInput],
    registry: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    max_budget: int,
    n_iterations: int,
) -> pd.DataFrame:
    variant_by_id = {variant.variant_id: variant for variant in variants}
    manifest_by_variant = _manifest_by_signature(manifest)
    rows: list[dict[str, Any]] = []
    for variant_id, group in registry.sort_values(["variant_id", "handle_candidate_id"]).groupby("variant_id", sort=True):
        variant = variant_by_id[str(variant_id)]
        graph = variant.graph
        node_names = list(map(str, graph.vs["name"]))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        candidates = list(group.itertuples(index=False))
        candidate_seen: dict[str, int] = defaultdict(int)
        for attempt_index in range(max_budget):
            candidate = candidates[attempt_index % len(candidates)]
            candidate_id = str(candidate.handle_candidate_id)
            method_seed = candidate_seen[candidate_id]
            candidate_seen[candidate_id] += 1
            groups = tuple(tuple(item) for item in json.loads(str(candidate.initial_groups)))
            initial = _initial_membership(node_names, groups)
            result = runner.run(variant.gamma, seed=method_seed, initial_membership=initial)
            membership = list(map(int, result.membership))
            endpoint_groups = _canonical_groups(graph, membership)
            signature_id = _signature_id(endpoint_groups)
            frozen_endpoint_id: str | None = None
            baseline_role: str | None = None
            endpoint_rank: int | None = None
            endpoint_mechanism_read: str | None = None
            if signature_id in manifest_by_variant[variant.variant_id].index:
                matched = manifest_by_variant[variant.variant_id].loc[signature_id]
                frozen_endpoint_id = str(matched["frozen_endpoint_id"])
                baseline_role = str(matched["baseline_role"])
                endpoint_rank = int(matched["endpoint_rank_in_variant"])
                endpoint_mechanism_read = str(matched["mechanism_read"])
            result_mechanism = _variant_mechanism_read(variant, membership)
            target_class_hit = bool(
                frozen_endpoint_id is not None
                and baseline_role == "recurrent_baseline_endpoint"
                and bool(candidate.target_claim_allowed)
                and _target_class_compatible(str(candidate.target_mechanism_class), endpoint_mechanism_read)
            )
            rows.append(
                {
                    "variant_id": variant.variant_id,
                    "mechanism_family": variant.mechanism_family,
                    "mechanism_state": variant.mechanism_state,
                    "responsible_rule": variant.responsible_rule,
                    "attempt_index": int(attempt_index + 1),
                    "handle_candidate_id": candidate_id,
                    "candidate_signature_id": str(candidate.candidate_signature_id),
                    "candidate_class_signature_id": str(candidate.candidate_class_signature_id),
                    "handle_type": str(candidate.handle_type),
                    "target_mechanism_class": str(candidate.target_mechanism_class),
                    "target_claim_allowed": bool(candidate.target_claim_allowed),
                    "method_seed": int(method_seed),
                    "gamma": float(variant.gamma),
                    "cluster_count": int(result.cluster_count),
                    "quality": float(result.quality),
                    "endpoint_signature_id": signature_id,
                    "endpoint_signature": json.dumps(endpoint_groups, sort_keys=True),
                    "frozen_endpoint_id": frozen_endpoint_id,
                    "baseline_role": baseline_role,
                    "endpoint_rank_in_variant": endpoint_rank,
                    "endpoint_mechanism_read": endpoint_mechanism_read,
                    "result_mechanism_read": result_mechanism,
                    "is_frozen_endpoint_hit": frozen_endpoint_id is not None,
                    "is_recurrent_endpoint_hit": baseline_role == "recurrent_baseline_endpoint",
                    "is_target_class_hit": target_class_hit,
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "attempt_index"]))


def _endpoint_first_hits(
    attempts: pd.DataFrame,
    manifest: pd.DataFrame,
    baseline_discovery: pd.DataFrame,
) -> pd.DataFrame:
    recurrent = manifest[manifest["is_recurrent_endpoint"].astype(bool)].copy()
    baseline_lookup = baseline_discovery.set_index(["variant_id", "frozen_endpoint_id"])
    rows: list[dict[str, Any]] = []
    for endpoint in recurrent.itertuples(index=False):
        variant_id = str(endpoint.variant_id)
        endpoint_id = str(endpoint.frozen_endpoint_id)
        hits = attempts[
            attempts["variant_id"].astype(str).eq(variant_id)
            & attempts["frozen_endpoint_id"].astype(str).eq(endpoint_id)
        ].copy()
        positive_hits = hits[hits["target_claim_allowed"].astype(bool)]
        target_class_hits = positive_hits[positive_hits["is_target_class_hit"].astype(bool)]
        first = hits.sort_values("attempt_index").head(1)
        first_positive = positive_hits.sort_values("attempt_index").head(1)
        first_target_class = target_class_hits.sort_values("attempt_index").head(1)
        baseline = baseline_lookup.loc[(variant_id, endpoint_id)]
        rows.append(
            {
                "variant_id": variant_id,
                "mechanism_family": str(endpoint.mechanism_family),
                "mechanism_state": str(endpoint.mechanism_state),
                "frozen_endpoint_id": endpoint_id,
                "endpoint_signature_id": str(endpoint.endpoint_signature_id),
                "mechanism_read": str(endpoint.mechanism_read),
                "seed_count": int(endpoint.seed_count),
                "seed_share": float(endpoint.seed_share),
                "first_attempt": int(first["attempt_index"].iloc[0]) if len(first) else None,
                "first_handle_candidate_id": str(first["handle_candidate_id"].iloc[0]) if len(first) else "",
                "first_handle_type": str(first["handle_type"].iloc[0]) if len(first) else "",
                "first_target_claim_allowed": bool(first["target_claim_allowed"].iloc[0]) if len(first) else False,
                "first_positive_attempt": int(first_positive["attempt_index"].iloc[0]) if len(first_positive) else None,
                "first_positive_handle_candidate_id": str(first_positive["handle_candidate_id"].iloc[0])
                if len(first_positive)
                else "",
                "first_target_class_attempt": int(first_target_class["attempt_index"].iloc[0])
                if len(first_target_class)
                else None,
                "first_target_class_handle_candidate_id": str(first_target_class["handle_candidate_id"].iloc[0])
                if len(first_target_class)
                else "",
                "baseline_first_hit_p75": float(baseline["baseline_first_hit_p75"]),
                "baseline_first_hit_p90": float(baseline["baseline_first_hit_p90"]),
                "any_candidate_hit": bool(len(first)),
                "positive_candidate_hit": bool(len(first_positive)),
                "target_class_candidate_hit": bool(len(first_target_class)),
                "positive_beats_restart_p75": bool(
                    len(first_positive) and int(first_positive["attempt_index"].iloc[0]) <= float(baseline["baseline_first_hit_p75"])
                ),
                "positive_beats_restart_p90": bool(
                    len(first_positive) and int(first_positive["attempt_index"].iloc[0]) <= float(baseline["baseline_first_hit_p90"])
                ),
                "target_class_beats_restart_p75": bool(
                    len(first_target_class)
                    and int(first_target_class["attempt_index"].iloc[0]) <= float(baseline["baseline_first_hit_p75"])
                ),
                "target_class_beats_restart_p90": bool(
                    len(first_target_class)
                    and int(first_target_class["attempt_index"].iloc[0]) <= float(baseline["baseline_first_hit_p90"])
                ),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "frozen_endpoint_id"]))


def _target_attribution(attempts: pd.DataFrame, first_hits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = attempts.groupby(["variant_id", "mechanism_family", "mechanism_state", "handle_type"], sort=True)
    for keys, group in grouped:
        variant_id, family, state, handle_type = keys
        positive = group[group["target_claim_allowed"].astype(bool)]
        control = group[~group["target_claim_allowed"].astype(bool)]
        first_variant = first_hits[first_hits["variant_id"].astype(str).eq(str(variant_id))]
        rows.append(
            {
                "variant_id": str(variant_id),
                "mechanism_family": str(family),
                "mechanism_state": str(state),
                "handle_type": str(handle_type),
                "attempt_count": int(len(group)),
                "positive_attempt_count": int(len(positive)),
                "control_attempt_count": int(len(control)),
                "recurrent_endpoint_hit_count": int(group["is_recurrent_endpoint_hit"].astype(bool).sum()),
                "positive_recurrent_endpoint_hit_count": int(positive["is_recurrent_endpoint_hit"].astype(bool).sum()),
                "target_class_endpoint_hit_count": int(group["is_target_class_hit"].astype(bool).sum()),
                "control_recurrent_endpoint_hit_count": int(control["is_recurrent_endpoint_hit"].astype(bool).sum()),
                "variant_recurrent_endpoint_count": int(len(first_variant)),
                "variant_positive_endpoint_hit_count": int(first_variant["positive_candidate_hit"].astype(bool).sum()),
                "variant_target_class_endpoint_hit_count": int(
                    first_variant["target_class_candidate_hit"].astype(bool).sum()
                ),
                "variant_positive_beats_restart_p75_count": int(
                    first_variant["positive_beats_restart_p75"].astype(bool).sum()
                ),
                "variant_target_class_beats_restart_p75_count": int(
                    first_variant["target_class_beats_restart_p75"].astype(bool).sum()
                ),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "handle_type"]))


def _gate_matrix(
    *,
    phase_lock_verified: bool,
    phase_lock_mismatches: list[str],
    p4_5_ready: bool,
    variants: list[VariantInput],
    endpoint_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    attempts: pd.DataFrame,
    first_hits: pd.DataFrame,
) -> pd.DataFrame:
    preserved_variants = [variant.variant_id for variant in variants if "control" not in variant.mechanism_state]
    controls = [variant.variant_id for variant in variants if "control" in variant.mechanism_state]
    recurrent_by_variant = (
        manifest[manifest["is_recurrent_endpoint"].astype(bool)]
        .groupby("variant_id")["frozen_endpoint_id"]
        .nunique()
        .to_dict()
    )
    preserved_multi = sum(1 for variant in preserved_variants if recurrent_by_variant.get(variant, 0) >= 2)
    preserved_with_recurrent = sum(1 for variant in preserved_variants if recurrent_by_variant.get(variant, 0) >= 1)
    control_positive_attempts = attempts[
        attempts["variant_id"].astype(str).isin(controls) & attempts["target_claim_allowed"].astype(bool)
    ]
    first_hit_preserved = first_hits[first_hits["variant_id"].astype(str).isin(preserved_variants)]
    positive_hit_count = int(first_hit_preserved["positive_candidate_hit"].astype(bool).sum())
    beats_p75_count = int(first_hit_preserved["positive_beats_restart_p75"].astype(bool).sum())
    target_class_hit_count = int(first_hit_preserved["target_class_candidate_hit"].astype(bool).sum())
    target_class_beats_p75_count = int(first_hit_preserved["target_class_beats_restart_p75"].astype(bool).sum())
    preserved_recurrent_endpoint_count = int(len(first_hit_preserved))
    target_class_by_variant = (
        first_hit_preserved.groupby("variant_id")["target_class_candidate_hit"].sum().to_dict()
        if not first_hit_preserved.empty
        else {}
    )
    zero_target_variants = sorted(
        variant for variant in preserved_variants if int(target_class_by_variant.get(variant, 0)) == 0
    )
    rows = [
        {
            "gate_id": "P5_G1_phase_lock_integrity",
            "gate_question": "Do P0-P4 artifacts still match the phase-lock before endpoint evaluation?",
            "status": "pass" if phase_lock_verified else "blocked_hash_mismatch",
            "evidence": "all locked hashes match" if phase_lock_verified else json.dumps(phase_lock_mismatches),
            "decision": "endpoint results are admissible only if pass",
        },
        {
            "gate_id": "P5_G2_p4_5_control_preflight",
            "gate_question": "Did the P4.5 control-strength audit clear the hard decoy gate?",
            "status": "pass" if p4_5_ready else "blocked_p4_5_not_ready",
            "evidence": f"p4_5_ready={p4_5_ready}",
            "decision": "do not interpret P5-P8 until P4.5 is ready",
        },
        {
            "gate_id": "P5_G3_baseline_endpoint_universe",
            "gate_question": "Did every variant produce at least one recurrent Leiden+CPM endpoint?",
            "status": "pass" if len(recurrent_by_variant) == len(variants) else "blocked_missing_recurrent_endpoint",
            "evidence": f"variants_with_recurrent_endpoint={len(recurrent_by_variant)}/{len(variants)}",
            "decision": "freeze recurrent endpoint universe if pass",
        },
        {
            "gate_id": "P6_G4_preserved_basin_diversity",
            "gate_question": "Do preserved variants show multiple recurrent endpoints?",
            "status": "pass" if preserved_multi >= 1 else "caveat_preserved_variants_single_endpoint",
            "evidence": f"preserved_multi_recurrent_variants={preserved_multi}/{len(preserved_variants)}",
            "decision": "supports basin-diversity read only if pass or carefully caveated",
        },
        {
            "gate_id": "P7_G5_no_control_positive_claims",
            "gate_question": "Are mechanism-removed controls still prevented from positive target attribution?",
            "status": "pass" if control_positive_attempts.empty else "blocked_control_positive_attempts",
            "evidence": f"control_positive_attempt_rows={len(control_positive_attempts)}",
            "decision": "controls cannot support target claims",
        },
        {
            "gate_id": "P7_G6_preserved_positive_hits",
            "gate_question": "Do positive candidates hit target-compatible recurrent endpoints in preserved variants?",
            "status": "pass" if target_class_hit_count > 0 else "caveat_no_target_class_recurrent_hits",
            "evidence": (
                f"preserved_with_recurrent={preserved_with_recurrent}, "
                f"positive_recurrent_endpoint_hits={positive_hit_count}, "
                f"target_class_recurrent_endpoint_hits={target_class_hit_count}"
            ),
            "decision": "candidate rules have target-scoped endpoint contact only if pass",
        },
        {
            "gate_id": "P8_G7_restart_p75_comparison",
            "gate_question": "Do target-compatible positive candidate first hits beat random-restart p75?",
            "status": "pass" if target_class_beats_p75_count > 0 else "caveat_no_target_class_restart_p75_gain",
            "evidence": (
                f"positive_beats_restart_p75_count={beats_p75_count}, "
                f"target_class_beats_restart_p75_count={target_class_beats_p75_count}"
            ),
            "decision": "cost-adjusted improvement remains diagnostic unless target-scoped coverage is broad",
        },
        {
            "gate_id": "P8_G8_target_scope_coverage",
            "gate_question": "Is target-compatible endpoint coverage complete across preserved recurrent endpoints?",
            "status": "pass"
            if target_class_hit_count == preserved_recurrent_endpoint_count
            else "caveat_incomplete_target_scoped_coverage",
            "evidence": (
                f"target_class_hits={target_class_hit_count}/{preserved_recurrent_endpoint_count}, "
                f"zero_target_hit_preserved_variants={json.dumps(zero_target_variants)}"
            ),
            "decision": "do not promote Stress 4 evidence while coverage misses remain",
        },
    ]
    _ = endpoint_summary
    return _with_claim_columns(pd.DataFrame(rows))


def _readiness(gates: pd.DataFrame) -> str:
    statuses = set(gates["status"].astype(str))
    if any(status.startswith("blocked") for status in statuses):
        return "blocked_fix_inputs_or_runner_before_interpretation"
    if any(status.startswith("caveat") for status in statuses):
        return "caveated_endpoint_diagnostic_only"
    return "ready_for_stress4_endpoint_diagnostic_review"


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows)
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in rows.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", r"\|"))
        table.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        table.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(table)


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    gates: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
    first_hits: pd.DataFrame,
    attribution: pd.DataFrame,
) -> None:
    lines = [
        "# Tiny CPM Mechanism Variant Panel P5-P8",
        "",
        f"- input_dir: `{summary['input_dir']}`",
        f"- audit_dir: `{summary['audit_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- panel_version: `{summary['panel_version']}`",
        f"- readiness: `{summary['p5_p8_readiness']}`",
        f"- variant_count: `{summary['variant_count']}`",
        f"- recurrent_endpoint_count: `{summary['recurrent_endpoint_count']}`",
        f"- preserved_recurrent_endpoint_count: `{summary['preserved_recurrent_endpoint_count']}`",
        f"- positive_recurrent_endpoint_hit_count: `{summary['positive_recurrent_endpoint_hit_count']}`",
        f"- positive_beats_restart_p75_count: `{summary['positive_beats_restart_p75_count']}`",
        f"- target_class_recurrent_endpoint_hit_count: `{summary['target_class_recurrent_endpoint_hit_count']}`",
        f"- target_class_beats_restart_p75_count: `{summary['target_class_beats_restart_p75_count']}`",
        f"- target_class_endpoint_hit_rate: `{summary['target_class_endpoint_hit_rate']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(gates, ["gate_id", "status", "evidence", "decision"], max_rows=12),
        "",
        "## Endpoint Summary",
        "",
        _markdown_table(
            endpoint_summary,
            [
                "variant_id",
                "mechanism_family",
                "mechanism_state",
                "mechanism_read",
                "seed_count",
                "seed_share",
                "is_recurrent_endpoint",
                "quality_median",
            ],
            max_rows=32,
        ),
        "",
        "## First Hits",
        "",
        _markdown_table(
            first_hits,
            [
                "variant_id",
                "frozen_endpoint_id",
                "mechanism_read",
                "first_positive_attempt",
                "first_target_class_attempt",
                "baseline_first_hit_p75",
                "positive_candidate_hit",
                "target_class_candidate_hit",
                "target_class_beats_restart_p75",
            ],
            max_rows=32,
        ),
        "",
        "## Target Attribution",
        "",
        _markdown_table(
            attribution,
            [
                "variant_id",
                "handle_type",
                "positive_attempt_count",
                "positive_recurrent_endpoint_hit_count",
                "target_class_endpoint_hit_count",
                "variant_positive_beats_restart_p75_count",
                "variant_target_class_beats_restart_p75_count",
            ],
            max_rows=32,
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- P5-P8 can evaluate endpoint contact and restart-position diagnostics.",
        "- This runner does not execute route/pathway traces or promote wall crossing.",
        "- Quality/cost, NanoClustering generality, and algorithm-level claims remain closed.",
    ]
    output_dir.joinpath(REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_p5_p8(
    *,
    input_dir: Path,
    audit_dir: Path,
    output_dir: Path,
    seeds: int | None,
    n_iterations: int,
    max_budget: int,
    baseline_permutations: int,
    discovery_rng_seed: int,
    require_p4_5_ready: bool,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_JSON
    if summary_path.exists() and not force:
        raise FileExistsError(f"{_rel(summary_path)} already exists. Use --force to regenerate P5-P8.")

    inputs = _load_inputs(input_dir, audit_dir)
    p4_5_ready = (
        inputs["audit_summary"].get("p5_p8_readiness") == "ready_for_p5_p8_diagnostic_execution"
        and bool(inputs["audit_summary"].get("hard_control_decoy_gate"))
        and int(inputs["audit_summary"].get("weak_control_count", 1)) == 0
        and int(inputs["audit_summary"].get("blocked_control_count", 1)) == 0
    )
    if require_p4_5_ready and not p4_5_ready:
        raise RuntimeError(f"P4.5 audit is not ready for P5-P8: {_rel(audit_dir)}")
    if require_p4_5_ready and not inputs["phase_lock_verified"]:
        raise RuntimeError(f"P0-P4 phase lock is not verified: {inputs['phase_lock_mismatches']}")

    manifest = inputs["manifest"]
    p0_config = inputs["phase_lock"].get("config", {})
    panel_version = str(p0_config.get("panel_version", "v1_1"))
    seed_count = int(seeds if seeds is not None else p0_config.get("seeds", 100))
    recurrent_threshold = max(
        int(p0_config.get("recurrent_threshold_min", 2)),
        math.ceil(seed_count * float(p0_config.get("recurrent_threshold_share", 0.05))),
    )
    variants = _variant_inputs(manifest, inputs["edges"], inputs["roles"])
    seed_runs = _run_baseline(variants, seeds=seed_count, n_iterations=n_iterations)
    endpoint_summary = _endpoint_summary(
        seed_runs,
        seeds=seed_count,
        recurrent_threshold=recurrent_threshold,
    )
    frozen_manifest = _frozen_endpoint_manifest(endpoint_summary, panel_version=panel_version)
    baseline_discovery = _baseline_endpoint_discovery(
        seed_runs,
        frozen_manifest,
        permutations=baseline_permutations,
        rng_seed=discovery_rng_seed,
    )
    attempts = _candidate_attempts(
        variants,
        inputs["registry"],
        frozen_manifest,
        max_budget=max_budget,
        n_iterations=n_iterations,
    )
    first_hits = _endpoint_first_hits(attempts, frozen_manifest, baseline_discovery)
    attribution = _target_attribution(attempts, first_hits)
    gates = _gate_matrix(
        phase_lock_verified=inputs["phase_lock_verified"],
        phase_lock_mismatches=inputs["phase_lock_mismatches"],
        p4_5_ready=p4_5_ready,
        variants=variants,
        endpoint_summary=endpoint_summary,
        manifest=frozen_manifest,
        attempts=attempts,
        first_hits=first_hits,
    )
    readiness = _readiness(gates)

    config = {
        "input_dir": _rel(input_dir),
        "audit_dir": _rel(audit_dir),
        "output_dir": _rel(output_dir),
        "phase_lock_hash": inputs["phase_lock"].get("phase_lock_hash"),
        "panel_version": panel_version,
        "phase_lock_verified": inputs["phase_lock_verified"],
        "phase_lock_mismatches": inputs["phase_lock_mismatches"],
        "p4_5_ready": p4_5_ready,
        "seeds": seed_count,
        "n_iterations": n_iterations,
        "max_budget": max_budget,
        "baseline_permutations": baseline_permutations,
        "discovery_rng_seed": discovery_rng_seed,
        "recurrent_threshold": recurrent_threshold,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output_dir.joinpath(CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(seed_runs, output_dir / SEED_RUNS_CSV)
    _write_csv(endpoint_summary, output_dir / ENDPOINT_SUMMARY_CSV)
    _write_csv(frozen_manifest, output_dir / FROZEN_ENDPOINT_MANIFEST_CSV)
    _write_csv(baseline_discovery, output_dir / BASELINE_DISCOVERY_CSV)
    _write_csv(attempts, output_dir / CANDIDATE_ATTEMPTS_CSV)
    _write_csv(first_hits, output_dir / ENDPOINT_FIRST_HITS_CSV)
    _write_csv(attribution, output_dir / TARGET_ATTRIBUTION_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)

    preserved_first_hits = first_hits[~first_hits["mechanism_state"].astype(str).str.contains("control")]
    preserved_recurrent_endpoint_count = int(len(preserved_first_hits))
    target_class_recurrent_endpoint_hit_count = int(
        preserved_first_hits["target_class_candidate_hit"].astype(bool).sum()
    )
    target_class_beats_restart_p75_count = int(
        preserved_first_hits["target_class_beats_restart_p75"].astype(bool).sum()
    )
    summary = {
        "input_dir": _rel(input_dir),
        "audit_dir": _rel(audit_dir),
        "output_dir": _rel(output_dir),
        "phase_lock_hash": inputs["phase_lock"].get("phase_lock_hash"),
        "panel_version": panel_version,
        "phase_lock_verified": inputs["phase_lock_verified"],
        "p4_5_ready": p4_5_ready,
        "variant_count": int(len(variants)),
        "seed_run_count": int(len(seed_runs)),
        "endpoint_count": int(len(endpoint_summary)),
        "recurrent_endpoint_count": int(frozen_manifest["is_recurrent_endpoint"].astype(bool).sum()),
        "preserved_recurrent_endpoint_count": preserved_recurrent_endpoint_count,
        "candidate_attempt_count": int(len(attempts)),
        "positive_recurrent_endpoint_hit_count": int(
            preserved_first_hits["positive_candidate_hit"].astype(bool).sum()
        ),
        "positive_beats_restart_p75_count": int(
            preserved_first_hits["positive_beats_restart_p75"].astype(bool).sum()
        ),
        "target_class_recurrent_endpoint_hit_count": target_class_recurrent_endpoint_hit_count,
        "target_class_beats_restart_p75_count": target_class_beats_restart_p75_count,
        "target_class_endpoint_hit_rate": (
            float(target_class_recurrent_endpoint_hit_count / preserved_recurrent_endpoint_count)
            if preserved_recurrent_endpoint_count
            else 0.0
        ),
        "p5_p8_readiness": readiness,
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            SEED_RUNS_CSV,
            ENDPOINT_SUMMARY_CSV,
            FROZEN_ENDPOINT_MANIFEST_CSV,
            BASELINE_DISCOVERY_CSV,
            CANDIDATE_ATTEMPTS_CSV,
            ENDPOINT_FIRST_HITS_CSV,
            TARGET_ATTRIBUTION_CSV,
            GATE_MATRIX_CSV,
            SUMMARY_JSON,
            CONFIG_JSON,
            REPORT_MD,
        ],
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(
        output_dir=output_dir,
        summary=summary,
        gates=gates,
        endpoint_summary=endpoint_summary,
        first_hits=first_hits,
        attribution=attribution,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stress 4 mechanism-variant panel P5-P8 from phase-locked P0-P4 inputs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--n-iterations", type=int, default=-1)
    parser.add_argument("--max-budget", type=int, default=20)
    parser.add_argument("--baseline-permutations", type=int, default=1000)
    parser.add_argument("--discovery-rng-seed", type=int, default=20260601)
    parser.add_argument("--allow-p4-5-not-ready", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = run_p5_p8(
        input_dir=args.input_dir,
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        n_iterations=args.n_iterations,
        max_budget=args.max_budget,
        baseline_permutations=args.baseline_permutations,
        discovery_rng_seed=args.discovery_rng_seed,
        require_p4_5_ready=not args.allow_p4_5_not_ready,
        force=args.force,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
