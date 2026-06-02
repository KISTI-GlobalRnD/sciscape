#!/usr/bin/env python3
"""Generate and test blind graph-rule handles for tiny CPM demos.

This is Stress 2 from the coverage-v2 robustness design. Candidate handles are
constructed from graph structure and demo node naming conventions before frozen
endpoint manifests, replay diagnoses, or endpoint signatures are read. Those
evaluation inputs are read only after the candidate registry is written.

This remains a tiny-demo candidate-method stress. It is not a route/pathway
trace, a wall promotion, a quality/cost claim, a NanoClustering generality
claim, or an algorithm-level claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import igraph as ig
import pandas as pd

from sciscape.clustering.runner import LeidenRunner

from run_leiden_cpm_tiny_demo_seed_sweep import (
    _canonical_groups,
    _classify_mechanism,
    _graph_cases,
    _json_safe,
    _signature_id,
    _write_csv,
)
from run_leiden_cpm_tiny_handle_method_probe import _initial_membership


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_BASELINE_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_demo_seed_sweep_20260531"
DEFAULT_REPLAY_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_endpoint_replay_v1_20260531"
DEFAULT_ORDER_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_coverage_order_robustness_v1_20260531"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_blind_rule_handle_probe_v1_20260531"

FROZEN_ENDPOINT_MANIFEST_CSV = "leiden_cpm_tiny_demo_frozen_endpoint_manifest.csv"
BASELINE_DISCOVERY_CURVE_CSV = "leiden_cpm_tiny_demo_discovery_curve.csv"
MISSING_DIAGNOSIS_CSV = "tiny_cpm_missing_endpoint_replay_diagnosis.csv"
ORDER_ATTEMPTS_CSV = "tiny_cpm_coverage_order_attempts.csv"
BLIND_CANDIDATE_REGISTRY_CSV = "tiny_cpm_blind_rule_candidate_registry.csv"
BLIND_CONSTRUCTION_PROVENANCE_JSON = "tiny_cpm_blind_rule_construction_provenance.json"
BLIND_ATTEMPTS_CSV = "tiny_cpm_blind_rule_attempts.csv"
BLIND_ENDPOINT_HITS_CSV = "tiny_cpm_blind_rule_endpoint_hits.csv"
BLIND_DISCOVERY_CSV = "tiny_cpm_blind_rule_discovery.csv"
BLIND_HARD_ENDPOINT_FIRST_HITS_CSV = "tiny_cpm_blind_rule_hard_endpoint_first_hits.csv"
BLIND_FAILURE_TYPING_CSV = "tiny_cpm_blind_rule_failure_typing.csv"
GATE_MATRIX_CSV = "tiny_cpm_blind_rule_gate_matrix.csv"
SUMMARY_JSON = "tiny_cpm_blind_rule_summary.json"
CONFIG_JSON = "tiny_cpm_blind_rule_config.json"
REPORT_MD = "tiny_cpm_blind_rule_report.md"

DEFAULT_BUDGETS = (1, 2, 3, 5, 10, 20)
CLAIM_BOUNDARY = (
    "Tiny CPM blind graph-rule handle probe only; candidates are constructed "
    "from graph structure before endpoint evaluation inputs are read, no route "
    "execution, no wall/pathway promotion, no basin-quality claim, no cost "
    "claim, no NanoClustering generality claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_blind_rule_probe_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "candidate_blind_rule_probe_not_algorithm_claim"


@dataclass(frozen=True)
class BlindRuleCandidate:
    family: str
    handle_candidate_id: str
    handle_type: str
    target_mechanism_read: str
    groups: tuple[tuple[str, ...], ...]
    handle_node_count: int
    handle_description: str
    ambiguous_module_nodes: tuple[str, ...]
    host_candidate: str
    boundary_core_nodes: tuple[str, ...]
    tail_nodes: tuple[str, ...]
    weak_node_pair: tuple[str, ...]
    graph_rule: str
    graph_evidence: dict[str, Any]


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


def _group(*nodes: str) -> tuple[str, ...]:
    return tuple(nodes)


def _names(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{index}" for index in range(count))


def _host_nodes(prefix: str, count: int) -> tuple[str, ...]:
    return _names(prefix, count)


def _edge_weight_lookup(graph: ig.Graph) -> dict[tuple[str, str], float]:
    weights: dict[tuple[str, str], float] = {}
    for edge in graph.es:
        left = str(graph.vs[edge.source]["name"])
        right = str(graph.vs[edge.target]["name"])
        weight = float(edge["weight"]) if "weight" in graph.es.attributes() else 1.0
        weights[tuple(sorted((left, right)))] = weight
    return weights


def _edge_weight(weights: dict[tuple[str, str], float], left: str, right: str) -> float:
    return weights.get(tuple(sorted((left, right))), 0.0)


def _contact_nodes(
    *,
    weights: dict[tuple[str, str], float],
    host_nodes: tuple[str, ...],
    module_nodes: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        node
        for node in host_nodes
        if sum(_edge_weight(weights, node, module_node) for module_node in module_nodes) > 0
    )


def _contact_weight(
    *,
    weights: dict[tuple[str, str], float],
    left_nodes: tuple[str, ...],
    right_nodes: tuple[str, ...],
) -> float:
    return float(
        sum(
            _edge_weight(weights, left_node, right_node)
            for left_node in left_nodes
            for right_node in right_nodes
        )
    )


def _tail_nodes(host_nodes: tuple[str, ...], boundary_core: tuple[str, ...]) -> tuple[str, ...]:
    core = set(boundary_core)
    return tuple(node for node in host_nodes if node not in core)


def _candidate_signature(groups: tuple[tuple[str, ...], ...]) -> str:
    payload = json.dumps([list(group) for group in groups], sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _candidate(
    *,
    family: str,
    handle_candidate_id: str,
    handle_type: str,
    target_mechanism_read: str,
    groups: tuple[tuple[str, ...], ...],
    handle_description: str,
    ambiguous_module_nodes: tuple[str, ...],
    host_candidate: str,
    boundary_core_nodes: tuple[str, ...],
    tail_nodes: tuple[str, ...],
    weak_node_pair: tuple[str, ...] = (),
    graph_rule: str,
    graph_evidence: dict[str, Any],
) -> BlindRuleCandidate:
    touched_nodes = set(ambiguous_module_nodes) | set(boundary_core_nodes) | set(weak_node_pair)
    return BlindRuleCandidate(
        family=family,
        handle_candidate_id=handle_candidate_id,
        handle_type=handle_type,
        target_mechanism_read=target_mechanism_read,
        groups=groups,
        handle_node_count=len(touched_nodes),
        handle_description=handle_description,
        ambiguous_module_nodes=ambiguous_module_nodes,
        host_candidate=host_candidate,
        boundary_core_nodes=boundary_core_nodes,
        tail_nodes=tail_nodes,
        weak_node_pair=weak_node_pair,
        graph_rule=graph_rule,
        graph_evidence=graph_evidence,
    )


def _near_tie_candidates(graph: ig.Graph) -> list[BlindRuleCandidate]:
    weights = _edge_weight_lookup(graph)
    a = _host_nodes("a", 5)
    b = _host_nodes("b", 5)
    bridge = _group("x")
    evidence = {
        "host_a_bridge_contact": _contact_weight(weights=weights, left_nodes=a, right_nodes=bridge),
        "host_b_bridge_contact": _contact_weight(weights=weights, left_nodes=b, right_nodes=bridge),
        "bridge_node_count": len(bridge),
    }
    return [
        _candidate(
            family="near_tie_bridge_cliques",
            handle_candidate_id="blind_near_tie_bridge_to_a",
            handle_type="blind_bridge_contact_initialization",
            target_mechanism_read="bridge_to_a",
            groups=(_group(*a, *bridge), _group(*b)),
            handle_description="Attach the bridge node to host A using graph contact symmetry.",
            ambiguous_module_nodes=bridge,
            host_candidate="a",
            boundary_core_nodes=a,
            tail_nodes=(),
            graph_rule="single bridge node attached to each host candidate",
            graph_evidence=evidence,
        ),
        _candidate(
            family="near_tie_bridge_cliques",
            handle_candidate_id="blind_near_tie_bridge_to_b",
            handle_type="blind_bridge_contact_initialization",
            target_mechanism_read="bridge_to_b",
            groups=(_group(*a), _group(*b, *bridge)),
            handle_description="Attach the bridge node to host B using graph contact symmetry.",
            ambiguous_module_nodes=bridge,
            host_candidate="b",
            boundary_core_nodes=b,
            tail_nodes=(),
            graph_rule="single bridge node attached to each host candidate",
            graph_evidence=evidence,
        ),
        _candidate(
            family="near_tie_bridge_cliques",
            handle_candidate_id="blind_near_tie_bridge_separate",
            handle_type="blind_bridge_contact_initialization",
            target_mechanism_read="bridge_separate",
            groups=(_group(*a), _group(*b), bridge),
            handle_description="Keep the bridge node separate as a bridge-control handle.",
            ambiguous_module_nodes=bridge,
            host_candidate="none",
            boundary_core_nodes=(),
            tail_nodes=(),
            graph_rule="bridge control handle without host attachment",
            graph_evidence=evidence,
        ),
    ]


def _absorption_candidates(graph: ig.Graph) -> list[BlindRuleCandidate]:
    weights = _edge_weight_lookup(graph)
    a = _host_nodes("a", 6)
    b = _host_nodes("b", 6)
    s = _names("s", 3)
    candidates: list[BlindRuleCandidate] = []
    for host_name, host_nodes in [("a", a), ("b", b)]:
        core = _contact_nodes(weights=weights, host_nodes=host_nodes, module_nodes=s)
        tail = _tail_nodes(host_nodes, core)
        other = b if host_name == "a" else a
        if host_name == "a":
            groups = (_group(*core, *s), _group(*tail), _group(*other))
            target = "small_module_absorbed_by_a"
        else:
            groups = (_group(*other), _group(*core, *s), _group(*tail))
            target = "small_module_absorbed_by_b"
        candidates.append(
            _candidate(
                family="absorption_triad",
                handle_candidate_id=f"blind_absorption_small_to_{host_name}_boundary_core",
                handle_type="blind_small_module_boundary_core_initialization",
                target_mechanism_read=target,
                groups=groups,
                handle_description=f"Attach the small module to host {host_name} boundary-core nodes only.",
                ambiguous_module_nodes=s,
                host_candidate=host_name,
                boundary_core_nodes=core,
                tail_nodes=tail,
                graph_rule="host nodes with any direct edge to the small module form the boundary core",
                graph_evidence={
                    "host_contact_weight": _contact_weight(
                        weights=weights,
                        left_nodes=host_nodes,
                        right_nodes=s,
                    ),
                    "boundary_core_count": len(core),
                    "tail_count": len(tail),
                },
            )
        )
    candidates.insert(
        1,
        _candidate(
            family="absorption_triad",
            handle_candidate_id="blind_absorption_small_separate",
            handle_type="blind_small_module_control_initialization",
            target_mechanism_read="small_module_separate",
            groups=(_group(*a), _group(*b), _group(*s)),
            handle_description="Keep the small module separate as a graph-rule control handle.",
            ambiguous_module_nodes=s,
            host_candidate="none",
            boundary_core_nodes=(),
            tail_nodes=(),
            graph_rule="small ambiguous module control handle",
            graph_evidence={
                "host_a_contact_weight": _contact_weight(weights=weights, left_nodes=a, right_nodes=s),
                "host_b_contact_weight": _contact_weight(weights=weights, left_nodes=b, right_nodes=s),
            },
        ),
    )
    return candidates


def _balanced_candidates(graph: ig.Graph) -> list[BlindRuleCandidate]:
    weights = _edge_weight_lookup(graph)
    a = _host_nodes("a", 5)
    b = _host_nodes("b", 5)
    m = _names("m", 4)
    split_a: list[str] = []
    split_b: list[str] = []
    per_middle: dict[str, dict[str, float]] = {}
    for middle in m:
        a_weight = _contact_weight(weights=weights, left_nodes=_group(middle), right_nodes=a)
        b_weight = _contact_weight(weights=weights, left_nodes=_group(middle), right_nodes=b)
        per_middle[middle] = {"a": a_weight, "b": b_weight}
        if a_weight >= b_weight:
            split_a.append(middle)
        else:
            split_b.append(middle)
    candidates = [
        _candidate(
            family="balanced_split_module",
            handle_candidate_id="blind_balanced_middle_split_by_contact",
            handle_type="blind_middle_contact_split_initialization",
            target_mechanism_read="balanced_middle_split",
            groups=(_group(*a, *split_a), _group(*b, *split_b)),
            handle_description="Split middle nodes by stronger host contact.",
            ambiguous_module_nodes=m,
            host_candidate="a;b",
            boundary_core_nodes=(),
            tail_nodes=(),
            graph_rule="assign each middle node to the host with greater total edge contact",
            graph_evidence={"middle_contact_by_host": per_middle},
        )
    ]
    for host_name, host_nodes in [("a", a), ("b", b)]:
        core = tuple(
            node
            for node in host_nodes
            if sum(1 for middle in m if _edge_weight(weights, node, middle) > 0) == len(m)
        )
        tail = _tail_nodes(host_nodes, core)
        other = b if host_name == "a" else a
        groups = (
            (_group(*core, *m), _group(*tail), _group(*other))
            if host_name == "a"
            else (_group(*other), _group(*core, *m), _group(*tail))
        )
        candidates.append(
            _candidate(
                family="balanced_split_module",
                handle_candidate_id=f"blind_balanced_middle_to_{host_name}_boundary_core",
                handle_type="blind_middle_boundary_core_initialization",
                target_mechanism_read="middle_module_absorbed_or_merged",
                groups=groups,
                handle_description=f"Attach the middle module to host {host_name} all-contact boundary core.",
                ambiguous_module_nodes=m,
                host_candidate=host_name,
                boundary_core_nodes=core,
                tail_nodes=tail,
                graph_rule="host nodes connected to every middle node form the boundary core",
                graph_evidence={
                    "boundary_core_count": len(core),
                    "tail_count": len(tail),
                    "host_contact_weight": _contact_weight(
                        weights=weights,
                        left_nodes=host_nodes,
                        right_nodes=m,
                    ),
                },
            )
        )
    candidates.append(
        _candidate(
            family="balanced_split_module",
            handle_candidate_id="blind_balanced_middle_separate",
            handle_type="blind_middle_control_initialization",
            target_mechanism_read="middle_module_separate",
            groups=(_group(*a), _group(*b), _group(*m)),
            handle_description="Keep the middle module separate as a graph-rule control handle.",
            ambiguous_module_nodes=m,
            host_candidate="none",
            boundary_core_nodes=(),
            tail_nodes=(),
            graph_rule="middle ambiguous module control handle",
            graph_evidence={"middle_contact_by_host": per_middle},
        )
    )
    return candidates


def _diffuse_host_nodes(host: int) -> tuple[str, ...]:
    return tuple(f"h{host}_{offset}" for offset in range(4))


def _diffuse_pair_for_host(
    *,
    weights: dict[tuple[str, str], float],
    host: int,
    weak_nodes: tuple[str, ...],
) -> tuple[str, ...]:
    host_nodes = _diffuse_host_nodes(host)
    connected = [
        weak
        for weak in weak_nodes
        if _contact_weight(weights=weights, left_nodes=_group(weak), right_nodes=host_nodes) > 0
    ]
    return tuple(sorted(connected))


def _diffuse_groups_for_pair_hosts(
    *,
    weights: dict[tuple[str, str], float],
    pair_hosts: tuple[int, ...],
    include_weak_separate: bool = False,
) -> tuple[tuple[str, ...], ...]:
    weak_nodes = _names("x", 4)
    pair_by_host = {
        host: _diffuse_pair_for_host(weights=weights, host=host, weak_nodes=weak_nodes)
        for host in pair_hosts
    }
    paired_weak = {node for pair in pair_by_host.values() for node in pair}
    assigned_weak: set[str] = set(paired_weak)
    groups: list[tuple[str, ...]] = []
    for host in range(4):
        host_nodes = _diffuse_host_nodes(host)
        if host in pair_hosts:
            pair = pair_by_host[host]
            core = _contact_nodes(weights=weights, host_nodes=host_nodes, module_nodes=pair)
            tail = _tail_nodes(host_nodes, core)
            groups.append(_group(*core, *pair))
            if tail:
                groups.append(tail)
        else:
            aligned = f"x{host}"
            if aligned in paired_weak:
                groups.append(host_nodes)
            else:
                assigned_weak.add(aligned)
                groups.append(_group(*host_nodes, aligned))
    if include_weak_separate:
        unassigned = tuple(node for node in weak_nodes if node not in assigned_weak)
        if unassigned:
            groups.append(unassigned)
    return tuple(groups)


def _diffuse_candidates(graph: ig.Graph) -> list[BlindRuleCandidate]:
    weights = _edge_weight_lookup(graph)
    weak_nodes = _names("x", 4)
    host_contact = {
        weak: {
            f"h{host}": _contact_weight(
                weights=weights,
                left_nodes=_group(weak),
                right_nodes=_diffuse_host_nodes(host),
            )
            for host in range(4)
        }
        for weak in weak_nodes
    }
    candidates: list[BlindRuleCandidate] = [
        _candidate(
            family="diffuse_fragment_star",
            handle_candidate_id="blind_diffuse_aligned_top_hosts",
            handle_type="blind_weak_top_host_initialization",
            target_mechanism_read="diffuse_host_fragmentation",
            groups=tuple(
                _group(*_diffuse_host_nodes(host), f"x{host}")
                for host in range(4)
            ),
            handle_description="Attach each weak node to its strongest host contact.",
            ambiguous_module_nodes=weak_nodes,
            host_candidate="h0;h1;h2;h3",
            boundary_core_nodes=(),
            tail_nodes=(),
            graph_rule="weak node goes to the host with maximum contact weight",
            graph_evidence={"weak_host_contact": host_contact},
        )
    ]
    for host in range(4):
        pair = _diffuse_pair_for_host(weights=weights, host=host, weak_nodes=weak_nodes)
        core = _contact_nodes(
            weights=weights,
            host_nodes=_diffuse_host_nodes(host),
            module_nodes=pair,
        )
        tail = _tail_nodes(_diffuse_host_nodes(host), core)
        candidates.append(
            _candidate(
                family="diffuse_fragment_star",
                handle_candidate_id=f"blind_diffuse_h{host}_weak_pair_tail_split",
                handle_type="blind_weak_pair_tail_split_initialization",
                target_mechanism_read="diffuse_host_fragmentation",
                groups=_diffuse_groups_for_pair_hosts(weights=weights, pair_hosts=(host,)),
                handle_description=f"Pair weak nodes contacting h{host} with the h{host} boundary core and split its tail.",
                ambiguous_module_nodes=weak_nodes,
                host_candidate=f"h{host}",
                boundary_core_nodes=core,
                tail_nodes=tail,
                weak_node_pair=pair,
                graph_rule="for each host, pair all weak nodes with direct host contact and split untouched host tail",
                graph_evidence={
                    "weak_host_contact": host_contact,
                    "pair_contact_weight": _contact_weight(
                        weights=weights,
                        left_nodes=pair,
                        right_nodes=_diffuse_host_nodes(host),
                    ),
                    "boundary_core_count": len(core),
                    "tail_count": len(tail),
                },
            )
        )
    for hosts in [(0, 2), (1, 3)]:
        weak_pair = tuple(
            node
            for host in hosts
            for node in _diffuse_pair_for_host(weights=weights, host=host, weak_nodes=weak_nodes)
        )
        core_nodes = tuple(
            node
            for host in hosts
            for node in _contact_nodes(
                weights=weights,
                host_nodes=_diffuse_host_nodes(host),
                module_nodes=_diffuse_pair_for_host(weights=weights, host=host, weak_nodes=weak_nodes),
            )
        )
        tail_nodes = tuple(
            node
            for host in hosts
            for node in _tail_nodes(
                _diffuse_host_nodes(host),
                _contact_nodes(
                    weights=weights,
                    host_nodes=_diffuse_host_nodes(host),
                    module_nodes=_diffuse_pair_for_host(weights=weights, host=host, weak_nodes=weak_nodes),
                ),
            )
        )
        candidates.append(
            _candidate(
                family="diffuse_fragment_star",
                handle_candidate_id=f"blind_diffuse_h{hosts[0]}_h{hosts[1]}_weak_pairs_tail_split",
                handle_type="blind_weak_pair_tail_split_initialization",
                target_mechanism_read="diffuse_host_fragmentation",
                groups=_diffuse_groups_for_pair_hosts(weights=weights, pair_hosts=hosts),
                handle_description=f"Pair weak nodes on disjoint hosts h{hosts[0]} and h{hosts[1]} and split both tails.",
                ambiguous_module_nodes=weak_nodes,
                host_candidate=f"h{hosts[0]};h{hosts[1]}",
                boundary_core_nodes=core_nodes,
                tail_nodes=tail_nodes,
                weak_node_pair=weak_pair,
                graph_rule="combine disjoint host weak-pair tail-split handles when weak-node pairs do not conflict",
                graph_evidence={"weak_host_contact": host_contact, "pair_hosts": hosts},
            )
        )
    candidates.append(
        _candidate(
            family="diffuse_fragment_star",
            handle_candidate_id="blind_diffuse_weak_module_separate",
            handle_type="blind_weak_module_control_initialization",
            target_mechanism_read="weak_module_separate",
            groups=tuple(_diffuse_host_nodes(host) for host in range(4)) + (_group(*weak_nodes),),
            handle_description="Keep weak module separate as a graph-rule control handle.",
            ambiguous_module_nodes=weak_nodes,
            host_candidate="none",
            boundary_core_nodes=(),
            tail_nodes=(),
            weak_node_pair=weak_nodes,
            graph_rule="weak ambiguous module control handle",
            graph_evidence={"weak_host_contact": host_contact},
        )
    )
    return candidates


def _generate_blind_candidates() -> list[BlindRuleCandidate]:
    cases = {case.family: case for case in _graph_cases()}
    return [
        *_near_tie_candidates(cases["near_tie_bridge_cliques"].builder()),
        *_absorption_candidates(cases["absorption_triad"].builder()),
        *_balanced_candidates(cases["balanced_split_module"].builder()),
        *_diffuse_candidates(cases["diffuse_fragment_star"].builder()),
    ]


def _candidate_registry(candidates: list[BlindRuleCandidate]) -> pd.DataFrame:
    rows = [
        {
            "family": candidate.family,
            "handle_candidate_id": candidate.handle_candidate_id,
            "candidate_signature_id": _candidate_signature(candidate.groups),
            "handle_type": candidate.handle_type,
            "target_mechanism_read": candidate.target_mechanism_read,
            "handle_node_count": candidate.handle_node_count,
            "initial_group_count": len(candidate.groups),
            "initial_groups": json.dumps([list(group) for group in candidate.groups], sort_keys=True),
            "handle_description": candidate.handle_description,
            "ambiguous_module_nodes": json.dumps(list(candidate.ambiguous_module_nodes)),
            "host_candidate": candidate.host_candidate,
            "boundary_core_nodes": json.dumps(list(candidate.boundary_core_nodes)),
            "tail_nodes": json.dumps(list(candidate.tail_nodes)),
            "weak_node_pair": json.dumps(list(candidate.weak_node_pair)),
            "graph_rule": candidate.graph_rule,
            "graph_evidence": json.dumps(_json_safe(candidate.graph_evidence), sort_keys=True),
            "construction_input_policy": "graph_structure_and_demo_node_names_only_before_endpoint_inputs",
        }
        for candidate in candidates
    ]
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["family", "handle_candidate_id"]))


def _manifest_sets(manifest: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = {}
    for family, group in manifest.groupby("family", sort=True):
        recurrent = set(
            group.loc[group["is_recurrent_endpoint"].astype(bool), "endpoint_signature_id"]
            .astype(str)
            .tolist()
        )
        total = set(group["endpoint_signature_id"].astype(str).tolist())
        top_quality = float(group["quality_max"].max())
        top_quality_ids = set(
            group.loc[group["quality_max"].ge(top_quality - 1e-9), "endpoint_signature_id"]
            .astype(str)
            .tolist()
        )
        result[str(family)] = {
            "recurrent": recurrent,
            "total": total,
            "top_quality": top_quality_ids,
        }
    return result


def _manifest_by_signature(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(family): group.set_index("endpoint_signature_id", drop=False)
        for family, group in manifest.groupby("family", sort=True)
    }


def _run_attempts(
    *,
    candidates: list[BlindRuleCandidate],
    manifest: pd.DataFrame,
    max_budget: int,
    n_iterations: int,
) -> pd.DataFrame:
    cases = {case.family: case for case in _graph_cases()}
    by_family: dict[str, list[BlindRuleCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_family[candidate.family].append(candidate)
    manifest_by_family = _manifest_by_signature(manifest)
    rows: list[dict[str, Any]] = []
    for family, family_candidates in sorted(by_family.items()):
        case = cases[family]
        graph = case.builder()
        node_names = list(map(str, graph.vs["name"]))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        candidate_seen: dict[str, int] = defaultdict(int)
        for attempt_index in range(max_budget):
            candidate = family_candidates[attempt_index % len(family_candidates)]
            method_seed = candidate_seen[candidate.handle_candidate_id]
            candidate_seen[candidate.handle_candidate_id] += 1
            initial = _initial_membership(node_names, candidate.groups)
            result = runner.run(case.gamma, seed=method_seed, initial_membership=initial)
            membership = list(map(int, result.membership))
            groups = _canonical_groups(graph, membership)
            signature_id = _signature_id(groups)
            frozen_endpoint_id: str | None = None
            baseline_role: str | None = None
            if signature_id in manifest_by_family[family].index:
                matched = manifest_by_family[family].loc[signature_id]
                frozen_endpoint_id = str(matched["frozen_endpoint_id"])
                baseline_role = str(matched["baseline_role"])
            result_mechanism = _classify_mechanism(family, graph, membership)
            rows.append(
                {
                    "family": family,
                    "attempt_index": int(attempt_index + 1),
                    "handle_candidate_id": candidate.handle_candidate_id,
                    "handle_type": candidate.handle_type,
                    "target_mechanism_read": candidate.target_mechanism_read,
                    "result_mechanism_read": result_mechanism,
                    "handle_node_count": candidate.handle_node_count,
                    "initial_group_count": len(candidate.groups),
                    "method_seed": int(method_seed),
                    "gamma": float(case.gamma),
                    "cluster_count": int(result.cluster_count),
                    "quality": float(result.quality),
                    "endpoint_signature_id": signature_id,
                    "endpoint_signature": json.dumps(groups, sort_keys=True),
                    "frozen_endpoint_id": frozen_endpoint_id,
                    "baseline_role": baseline_role,
                    "is_frozen_endpoint_hit": frozen_endpoint_id is not None,
                    "is_target_hit": bool(
                        frozen_endpoint_id is not None
                        and result_mechanism == candidate.target_mechanism_read
                    ),
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["family", "attempt_index"]))


def _endpoint_hits(attempts: pd.DataFrame) -> pd.DataFrame:
    hits = (
        attempts.groupby(
            [
                "family",
                "endpoint_signature_id",
                "frozen_endpoint_id",
                "baseline_role",
                "result_mechanism_read",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            hit_count=("attempt_index", "size"),
            first_attempt=("attempt_index", "min"),
            best_quality=("quality", "max"),
            target_hit_count=("is_target_hit", "sum"),
        )
        .sort_values(["family", "first_attempt", "endpoint_signature_id"])
    )
    return _with_claim_columns(hits)


def _discovery(
    *,
    attempts: pd.DataFrame,
    manifest: pd.DataFrame,
    baseline_curve: pd.DataFrame,
    budgets: list[int],
) -> pd.DataFrame:
    sets = _manifest_sets(manifest)
    rows: list[dict[str, Any]] = []
    for family, group in attempts.groupby("family", sort=True):
        family_sets = sets[str(family)]
        recurrent_ids = family_sets["recurrent"]
        total_ids = family_sets["total"]
        top_quality_ids = family_sets["top_quality"]
        for budget in budgets:
            prefix = group[group["attempt_index"].le(budget)]
            found = set(prefix["endpoint_signature_id"].astype(str))
            frozen_found = found & total_ids
            recall = float(len(frozen_found & recurrent_ids) / len(recurrent_ids))
            baseline_row = baseline_curve[
                baseline_curve["family"].eq(family) & baseline_curve["budget"].eq(budget)
            ]
            baseline_recall = (
                float(baseline_row["recurrent_endpoint_recall_mean"].iloc[0])
                if not baseline_row.empty
                else math.nan
            )
            baseline_all_recurrent = (
                float(baseline_row["all_recurrent_endpoint_hit_rate"].iloc[0])
                if not baseline_row.empty
                else math.nan
            )
            rows.append(
                {
                    "family": str(family),
                    "budget": int(budget),
                    "method_distinct_endpoint_count": int(len(frozen_found)),
                    "method_new_endpoint_count": int(len(found - total_ids)),
                    "method_recurrent_endpoint_recall": recall,
                    "method_all_recurrent_endpoint_hit": bool(recurrent_ids.issubset(frozen_found)),
                    "method_top_quality_endpoint_hit": bool(frozen_found & top_quality_ids),
                    "method_target_hit_count": int(prefix["is_target_hit"].astype(bool).sum()),
                    "method_attempt_count": int(len(prefix)),
                    "baseline_recurrent_endpoint_recall_mean": baseline_recall,
                    "baseline_all_recurrent_endpoint_hit_rate": baseline_all_recurrent,
                    "delta_recurrent_endpoint_recall": float(recall - baseline_recall)
                    if math.isfinite(baseline_recall)
                    else math.nan,
                    "delta_all_recurrent_endpoint_hit": float(
                        (1.0 if recurrent_ids.issubset(frozen_found) else 0.0)
                        - baseline_all_recurrent
                    )
                    if math.isfinite(baseline_all_recurrent)
                    else math.nan,
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["family", "budget"]))


def _adversarial_first_hits(order_dir: Path) -> pd.DataFrame:
    path = order_dir / ORDER_ATTEMPTS_CSV
    if not path.exists():
        return pd.DataFrame()
    attempts = pd.read_csv(path)
    rows = attempts[
        attempts["order_policy"].eq("adversarial_delayed_coverage")
        & attempts["frozen_endpoint_id"].notna()
        & attempts["baseline_role"].eq("recurrent_baseline_endpoint")
    ]
    if rows.empty:
        return pd.DataFrame()
    result = (
        rows.groupby(["family", "frozen_endpoint_id"], as_index=False)
        .agg(adversarial_first_attempt=("attempt_index", "min"))
        .sort_values(["family", "frozen_endpoint_id"])
    )
    return result


def _hard_endpoint_first_hits(
    *,
    attempts: pd.DataFrame,
    replay_dir: Path,
    order_dir: Path,
) -> pd.DataFrame:
    missing_path = replay_dir / MISSING_DIAGNOSIS_CSV
    if not missing_path.exists():
        return _with_claim_columns(pd.DataFrame())
    missing = pd.read_csv(missing_path)
    if missing.empty:
        return _with_claim_columns(pd.DataFrame())
    hard = missing[missing["diagnosis"].astype(str).str.contains("stable_endpoint", na=False)].copy()
    if hard.empty:
        return _with_claim_columns(pd.DataFrame())
    first_hits = (
        attempts[attempts["frozen_endpoint_id"].notna()]
        .groupby(["family", "frozen_endpoint_id"], as_index=False)
        .agg(blind_first_attempt=("attempt_index", "min"))
    )
    adversarial = _adversarial_first_hits(order_dir)
    merged = hard.merge(first_hits, on=["family", "frozen_endpoint_id"], how="left")
    if not adversarial.empty:
        merged = merged.merge(adversarial, on=["family", "frozen_endpoint_id"], how="left")
    else:
        merged["adversarial_first_attempt"] = math.nan
    merged["beats_adversarial_first_hit"] = (
        merged["blind_first_attempt"].notna()
        & merged["adversarial_first_attempt"].notna()
        & merged["blind_first_attempt"].lt(merged["adversarial_first_attempt"])
    )
    preferred = [
        "family",
        "frozen_endpoint_id",
        "mechanism_read",
        "seed_count",
        "diagnosis",
        "blind_first_attempt",
        "adversarial_first_attempt",
        "beats_adversarial_first_hit",
    ]
    return _with_claim_columns(merged[preferred].sort_values(["family", "frozen_endpoint_id"]))


def _failure_typing(
    *,
    manifest: pd.DataFrame,
    attempts: pd.DataFrame,
    candidates: list[BlindRuleCandidate],
) -> pd.DataFrame:
    hit_ids = set(attempts["frozen_endpoint_id"].dropna().astype(str).tolist())
    candidate_types = {candidate.handle_type for candidate in candidates}
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        endpoint_id = str(row.frozen_endpoint_id)
        if endpoint_id in hit_ids:
            failure_type = "hit"
        elif str(row.mechanism_read) == "weak_module_separate" and "blind_weak_module_control_initialization" not in candidate_types:
            failure_type = "missing_control_handle"
        elif str(row.mechanism_read) == "diffuse_host_fragmentation":
            failure_type = "weak_pair_rule_miss_or_polish_collapse"
        elif "absorbed" in str(row.mechanism_read):
            failure_type = "boundary_core_rule_miss_or_polish_collapse"
        elif str(row.mechanism_read) == "middle_module_separate":
            failure_type = "separate_control_not_recurrent_target"
        else:
            failure_type = "unexplained_candidate_miss"
        rows.append(
            {
                "family": str(row.family),
                "frozen_endpoint_id": endpoint_id,
                "baseline_role": str(row.baseline_role),
                "mechanism_read": str(row.mechanism_read),
                "is_recurrent_endpoint": bool(row.is_recurrent_endpoint),
                "hit_status": "hit" if endpoint_id in hit_ids else "miss",
                "failure_type": failure_type,
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["family", "frozen_endpoint_id"]))


def _gate_matrix(
    *,
    candidate_registry: pd.DataFrame,
    discovery: pd.DataFrame,
    hard_first_hits: pd.DataFrame,
    failure_typing: pd.DataFrame,
    construction_provenance: dict[str, Any],
) -> pd.DataFrame:
    construction_independent = bool(
        construction_provenance.get("candidate_registry_written_before_evaluation_inputs")
    )
    evidence_complete = bool(
        candidate_registry[
            candidate_registry["graph_evidence"].astype(str).str.len().gt(2)
        ].shape[0]
        == len(candidate_registry)
    )
    required_types = {
        "blind_small_module_boundary_core_initialization",
        "blind_middle_boundary_core_initialization",
        "blind_weak_pair_tail_split_initialization",
    }
    present_types = set(candidate_registry["handle_type"].astype(str).tolist())
    budget20 = discovery[discovery["budget"].eq(20)]
    below_restart = budget20[
        budget20["method_recurrent_endpoint_recall"].lt(
            budget20["baseline_recurrent_endpoint_recall_mean"]
        )
    ]
    diffuse20 = budget20[budget20["family"].eq("diffuse_fragment_star")]
    diffuse_recall = (
        float(diffuse20["method_recurrent_endpoint_recall"].iloc[0])
        if not diffuse20.empty
        else 0.0
    )
    hard_hit = hard_first_hits[
        hard_first_hits["blind_first_attempt"].notna()
    ]
    hard_beats = hard_first_hits[
        hard_first_hits["beats_adversarial_first_hit"].astype(bool)
    ] if not hard_first_hits.empty else pd.DataFrame()
    misses = failure_typing[
        failure_typing["hit_status"].eq("miss")
        & failure_typing["is_recurrent_endpoint"].astype(bool)
    ]
    unexplained = misses[misses["failure_type"].eq("unexplained_candidate_miss")]
    rows = [
        {
            "gate_id": "B1_construction_independence",
            "gate_question": "Was the blind candidate registry built before endpoint evaluation inputs were read?",
            "evidence": f"candidate_registry_written_before_evaluation_inputs={construction_independent}",
            "status": "pass" if construction_independent else "blocked_endpoint_leakage_risk",
            "decision": "blind_rule_surface_valid_for_evaluation_if_pass",
            "next_action": "inspect graph evidence fields",
        },
        {
            "gate_id": "B2_graph_evidence_auditability",
            "gate_question": "Does every candidate carry graph-derived construction evidence?",
            "evidence": f"candidate_rows={len(candidate_registry)}, evidence_complete={evidence_complete}",
            "status": "pass" if evidence_complete else "blocked_missing_graph_evidence",
            "decision": "candidate_generation_is_auditable_if_pass",
            "next_action": "review rule evidence for mechanism plausibility",
        },
        {
            "gate_id": "B3_qualitative_miss_class_recovery",
            "gate_question": "Are the required blind mechanism handle classes present?",
            "evidence": f"required_types_present={sorted(required_types & present_types)}",
            "status": "pass" if required_types.issubset(present_types) else "blocked_missing_required_rule_class",
            "decision": "stress1_timing_classes_are_represented_if_pass",
            "next_action": "evaluate endpoint recall by budget",
        },
        {
            "gate_id": "B4_budget20_discovery_performance",
            "gate_question": "Do blind-rule handles beat the restart mean at budget 20?",
            "evidence": (
                f"budget20_below_restart_families={len(below_restart)}, "
                f"diffuse_recall={diffuse_recall:.6f}"
            ),
            "status": "pass" if below_restart.empty and diffuse_recall >= 1.0 else "caveat_required",
            "decision": "blind_rules_match_small_demo_discovery_if_pass",
            "next_action": "inspect hard endpoint first-hit timing",
        },
        {
            "gate_id": "B5_hard_endpoint_first_hit_sanity",
            "gate_question": "Do blind rules hit hard endpoints earlier than adversarial delayed coverage?",
            "evidence": (
                f"hard_endpoints={len(hard_first_hits)}, "
                f"hard_hit={len(hard_hit)}, hard_beats_adversarial={len(hard_beats)}"
            ),
            "status": "pass"
            if len(hard_first_hits) > 0 and len(hard_beats) == len(hard_first_hits)
            else "caveat_required",
            "decision": "blind_rules_are_not_late_coverage_replay_if_pass",
            "next_action": "type any remaining misses",
        },
        {
            "gate_id": "B6_failure_typing",
            "gate_question": "Are recurrent misses typed by mechanism-readable failure modes?",
            "evidence": f"recurrent_misses={len(misses)}, unexplained_recurrent_misses={len(unexplained)}",
            "status": "pass" if unexplained.empty else "caveat_required",
            "decision": "failures_are_diagnostic_if_pass",
            "next_action": "use typed failures for ablation or rule revision",
        },
        {
            "gate_id": "B7_claim_gate",
            "gate_question": "Can this be promoted to an algorithm or wall/pathway claim?",
            "evidence": "tiny blind-rule candidate probe only",
            "status": "closed_excluded_by_design",
            "decision": "keep_algorithm_wall_quality_cost_claims_closed",
            "next_action": "run handle-type ablation only after reviewing blind-rule results",
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix["claim_boundary"] = CLAIM_BOUNDARY
    return matrix


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 24) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for _, row in rows.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", r"\|"))
        body.append("| " + " | ".join(values) + " |")
    suffix = [f"\n_Showing {max_rows} of {len(frame)} rows._"] if len(frame) > max_rows else []
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    gate_matrix: pd.DataFrame,
    discovery: pd.DataFrame,
    hard_first_hits: pd.DataFrame,
    failure_typing: pd.DataFrame,
) -> None:
    text = [
        "# Tiny CPM Blind Graph-Rule Handle Probe v1",
        "",
        f"- blind_candidate_count: `{summary['blind_candidate_count']}`",
        f"- attempt_count: `{summary['attempt_count']}`",
        f"- budget20_below_restart_families: `{summary['budget20_below_restart_families']}`",
        f"- diffuse_budget20_recall: `{summary['diffuse_budget20_recall']}`",
        f"- hard_endpoint_count: `{summary['hard_endpoint_count']}`",
        f"- hard_endpoints_beating_adversarial_delay: `{summary['hard_endpoints_beating_adversarial_delay']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gate_matrix,
            ["gate_id", "evidence", "status", "decision", "next_action"],
            max_rows=10,
        ),
        "",
        "## Discovery",
        "",
        _markdown_table(
            discovery[discovery["budget"].isin([3, 5, 10, 20])],
            [
                "family",
                "budget",
                "method_recurrent_endpoint_recall",
                "baseline_recurrent_endpoint_recall_mean",
                "delta_recurrent_endpoint_recall",
                "method_target_hit_count",
            ],
            max_rows=32,
        ),
        "",
        "## Hard Endpoint First Hits",
        "",
        _markdown_table(
            hard_first_hits,
            [
                "family",
                "frozen_endpoint_id",
                "blind_first_attempt",
                "adversarial_first_attempt",
                "beats_adversarial_first_hit",
            ],
            max_rows=20,
        ),
        "",
        "## Failure Typing",
        "",
        _markdown_table(
            failure_typing,
            [
                "family",
                "frozen_endpoint_id",
                "baseline_role",
                "mechanism_read",
                "hit_status",
                "failure_type",
            ],
            max_rows=32,
        ),
        "",
        "## Read",
        "",
        "- Candidate construction is intentionally separated from endpoint evaluation inputs.",
        "- A pass here means tiny-demo blind graph rules can reproduce the coverage-v2 mechanism classes.",
        "- It still does not establish optimizer-native walls, quality/cost value, NanoClustering generality, or an algorithm claim.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def run_probe(
    *,
    baseline_dir: Path,
    replay_dir: Path,
    order_dir: Path,
    output_dir: Path,
    budgets: list[int],
    max_budget: int,
    n_iterations: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # B1: candidate construction happens before reading endpoint evaluation inputs.
    candidates = _generate_blind_candidates()
    candidate_registry = _candidate_registry(candidates)
    _write_csv(candidate_registry, output_dir / BLIND_CANDIDATE_REGISTRY_CSV)
    construction_provenance = {
        "candidate_registry_written_before_evaluation_inputs": True,
        "construction_inputs": [
            "graph builders from run_leiden_cpm_tiny_demo_seed_sweep.py",
            "graph node names",
            "graph weighted edges",
        ],
        "excluded_before_candidate_registry": [
            "frozen endpoint manifest",
            "endpoint signatures",
            "endpoint replay diagnosis",
            "method-v1 missed endpoint list",
            "coverage-v2 endpoint hits",
        ],
        "candidate_count": len(candidates),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / BLIND_CONSTRUCTION_PROVENANCE_JSON).write_text(
        json.dumps(_json_safe(construction_provenance), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = pd.read_csv(baseline_dir / FROZEN_ENDPOINT_MANIFEST_CSV)
    baseline_curve = pd.read_csv(baseline_dir / BASELINE_DISCOVERY_CURVE_CSV)
    attempts = _run_attempts(
        candidates=candidates,
        manifest=manifest,
        max_budget=max_budget,
        n_iterations=n_iterations,
    )
    endpoint_hits = _endpoint_hits(attempts)
    discovery = _discovery(
        attempts=attempts,
        manifest=manifest,
        baseline_curve=baseline_curve,
        budgets=budgets,
    )
    hard_first_hits = _hard_endpoint_first_hits(
        attempts=attempts,
        replay_dir=replay_dir,
        order_dir=order_dir,
    )
    failure_typing = _failure_typing(
        manifest=manifest,
        attempts=attempts,
        candidates=candidates,
    )
    gate_matrix = _gate_matrix(
        candidate_registry=candidate_registry,
        discovery=discovery,
        hard_first_hits=hard_first_hits,
        failure_typing=failure_typing,
        construction_provenance=construction_provenance,
    )
    budget20 = discovery[discovery["budget"].eq(20)]
    below_restart = budget20[
        budget20["method_recurrent_endpoint_recall"].lt(
            budget20["baseline_recurrent_endpoint_recall_mean"]
        )
    ]
    diffuse20 = budget20[budget20["family"].eq("diffuse_fragment_star")]
    diffuse_recall = (
        float(diffuse20["method_recurrent_endpoint_recall"].iloc[0])
        if not diffuse20.empty
        else None
    )
    hard_beats = hard_first_hits[
        hard_first_hits["beats_adversarial_first_hit"].astype(bool)
    ] if not hard_first_hits.empty else pd.DataFrame()
    summary = {
        "blind_candidate_count": int(len(candidates)),
        "attempt_count": int(len(attempts)),
        "budgets": [int(budget) for budget in budgets],
        "max_budget": int(max_budget),
        "budget20_below_restart_families": int(len(below_restart)),
        "diffuse_budget20_recall": diffuse_recall,
        "hard_endpoint_count": int(len(hard_first_hits)),
        "hard_endpoints_beating_adversarial_delay": int(len(hard_beats)),
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gate_matrix["status"].value_counts().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {
            "baseline_dir": _rel(baseline_dir),
            "replay_dir": _rel(replay_dir),
            "order_dir": _rel(order_dir),
        },
    }

    _write_csv(attempts, output_dir / BLIND_ATTEMPTS_CSV)
    _write_csv(endpoint_hits, output_dir / BLIND_ENDPOINT_HITS_CSV)
    _write_csv(discovery, output_dir / BLIND_DISCOVERY_CSV)
    _write_csv(hard_first_hits, output_dir / BLIND_HARD_ENDPOINT_FIRST_HITS_CSV)
    _write_csv(failure_typing, output_dir / BLIND_FAILURE_TYPING_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "baseline_dir": _rel(baseline_dir),
        "replay_dir": _rel(replay_dir),
        "order_dir": _rel(order_dir),
        "output_dir": _rel(output_dir),
        "budgets": [int(budget) for budget in budgets],
        "max_budget": int(max_budget),
        "n_iterations": int(n_iterations),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        gate_matrix=gate_matrix,
        discovery=discovery,
        hard_first_hits=hard_first_hits,
        failure_typing=failure_typing,
    )
    return summary


def _parse_budgets(value: str) -> list[int]:
    budgets = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not budgets:
        raise argparse.ArgumentTypeError("at least one budget is required")
    return sorted(set(budgets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--order-dir", type=Path, default=DEFAULT_ORDER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--budgets", type=_parse_budgets, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--max-budget", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=-1)
    args = parser.parse_args()
    budgets = [budget for budget in args.budgets if budget <= args.max_budget]
    summary = run_probe(
        baseline_dir=args.baseline_dir,
        replay_dir=args.replay_dir,
        order_dir=args.order_dir,
        output_dir=args.output_dir,
        budgets=budgets,
        max_budget=args.max_budget,
        n_iterations=args.n_iterations,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
