#!/usr/bin/env python3
"""Materialize the Stress 4 tiny CPM mechanism-variant panel through P4.

This runner intentionally stops before any Leiden seed sweep or endpoint
evaluation. It writes graph manifests, graph-only mechanism diagnostics, blind
candidate registries, role/name invariance checks, and a phase-lock hash. The
next runner phase must treat these P0-P4 artifacts as frozen inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import igraph as ig
import pandas as pd

from run_leiden_cpm_tiny_demo_seed_sweep import (
    _add_clique,
    _add_edge,
    _build_graph,
    _json_safe,
    _write_csv,
)


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_OUTPUT_DIR_V1 = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_v1_20260531"
DEFAULT_OUTPUT_DIR_V1_1 = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_v1_1_20260601"
DEFAULT_OUTPUT_DIR_V1_2 = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_v1_2_20260601"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR_V1

GRAPH_MANIFEST_CSV = "tiny_cpm_variant_graph_manifest.csv"
GRAPH_EDGES_CSV = "tiny_cpm_variant_graph_edges.csv"
GRAPH_ROLES_CSV = "tiny_cpm_variant_graph_roles.csv"
MECHANISM_FEATURES_CSV = "tiny_cpm_variant_mechanism_features.csv"
ROLE_INVARIANCE_CSV = "tiny_cpm_variant_role_invariance.csv"
PHASE_LOCK_JSON = "tiny_cpm_variant_phase_lock.json"
BLIND_CANDIDATE_REGISTRY_CSV = "tiny_cpm_variant_blind_candidate_registry.csv"
CONFIG_JSON = "tiny_cpm_variant_config.json"
REPORT_MD = "tiny_cpm_variant_p0_p4_report.md"

CLAIM_BOUNDARY = (
    "Tiny CPM mechanism-variant panel P0-P4 only; graph manifest, graph-only "
    "diagnostics, blind candidate construction, role/name invariance, and "
    "phase lock before endpoint evaluation. No Leiden seed sweep, no endpoint "
    "manifest, no route/pathway execution, no wall promotion, no quality/cost "
    "claim, no NanoClustering generality claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_p0_p4_graph_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
METHOD_STATUS = "candidate_registry_phase_lock_only"

EXCLUDED_BEFORE_PHASE_LOCK = (
    "frozen_endpoint_manifest",
    "endpoint_replay_rows",
    "endpoint_ranks",
    "method_hit_rows",
    "seed_run_outcomes",
    "restart_discovery_curves",
)


@dataclass(frozen=True)
class RoleAnnotation:
    role_id: str
    node_ids: tuple[str, ...]
    role_type: str
    mechanism_family: str
    role_slot: str
    allowed_for_candidate_generation: bool = True


@dataclass(frozen=True)
class VariantGraphCase:
    variant_id: str
    mechanism_family: str
    mechanism_state: str
    gamma: float
    seed_count: int
    expected_baseline_behavior: str
    responsible_rule: str
    control_read: str
    builder: Callable[[], tuple[ig.Graph, tuple[RoleAnnotation, ...]]]
    panel_version: str = "v1"


@dataclass(frozen=True)
class GraphBundle:
    case: VariantGraphCase
    canonical_graph: ig.Graph
    roles: tuple[RoleAnnotation, ...]
    opaque_graph: ig.Graph
    opaque_roles: tuple[RoleAnnotation, ...]
    role_name_permuted_roles: tuple[RoleAnnotation, ...]
    node_id_map: dict[str, str]
    graph_hash: str
    role_hash: str


@dataclass(frozen=True)
class VariantCandidate:
    variant_id: str
    handle_candidate_id: str
    handle_type: str
    target_mechanism_class: str
    target_claim_allowed: bool
    initial_groups: tuple[tuple[str, ...], ...]
    initial_role_groups: tuple[tuple[str, ...], ...]
    touched_nodes: tuple[str, ...]
    evidence_role_ids: tuple[str, ...]
    graph_rule: str
    graph_evidence: dict[str, Any]
    candidate_signature: str
    candidate_class_signature: str


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


def _sha256_json(payload: Any, length: int | None = 16) -> str:
    text = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if length is None:
        return digest
    return digest[:length]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group(*nodes: str) -> tuple[str, ...]:
    return tuple(nodes)


def _nodes(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{index}" for index in range(count))


def _add_complete_bipartite(
    edges: dict[tuple[str, str], float],
    left_nodes: tuple[str, ...],
    right_nodes: tuple[str, ...],
    weight: float = 1.0,
) -> None:
    for left in left_nodes:
        for right in right_nodes:
            _add_edge(edges, left, right, weight)


def _graph_from_parts(
    names: tuple[str, ...],
    edges: dict[tuple[str, str], float],
) -> ig.Graph:
    return _build_graph(list(names), edges)


def _role(
    role_id: str,
    node_ids: tuple[str, ...],
    role_type: str,
    mechanism_family: str,
    role_slot: str,
    allowed: bool = True,
) -> RoleAnnotation:
    return RoleAnnotation(
        role_id=role_id,
        node_ids=tuple(node_ids),
        role_type=role_type,
        mechanism_family=mechanism_family,
        role_slot=role_slot,
        allowed_for_candidate_generation=allowed,
    )


def _near_tie_graph(
    variant_id: str,
    *,
    a_contact: int,
    b_contact: int,
    a_weight: float = 1.0,
    b_weight: float = 1.0,
) -> tuple[ig.Graph, tuple[RoleAnnotation, ...]]:
    family = "near_tie_bridge"
    a = _nodes(f"{variant_id}_a", 5)
    b = _nodes(f"{variant_id}_b", 5)
    bridge = _group(f"{variant_id}_x0")
    names = (*a, *b, *bridge)
    edges: dict[tuple[str, str], float] = defaultdict(float)
    _add_clique(edges, list(a))
    _add_clique(edges, list(b))
    for node in a[:a_contact]:
        _add_edge(edges, bridge[0], node, a_weight)
    for node in b[:b_contact]:
        _add_edge(edges, bridge[0], node, b_weight)
    roles = (
        _role(f"{variant_id}:host_a", a, "host", family, "host_a"),
        _role(f"{variant_id}:host_b", b, "host", family, "host_b"),
        _role(f"{variant_id}:bridge", bridge, "bridge", family, "bridge"),
    )
    return _graph_from_parts(names, edges), roles


def _absorption_graph(
    variant_id: str,
    *,
    mode: str,
) -> tuple[ig.Graph, tuple[RoleAnnotation, ...]]:
    family = "absorption_triad"
    a = _nodes(f"{variant_id}_a", 6)
    b = _nodes(f"{variant_id}_b", 6)
    s = _nodes(f"{variant_id}_s", 3)
    names = (*a, *b, *s)
    edges: dict[tuple[str, str], float] = defaultdict(float)
    _add_clique(edges, list(a))
    _add_clique(edges, list(b))
    _add_clique(edges, list(s))
    if mode == "boundary_vs_diffuse":
        a_core = a[:3]
        b_core = ()
        _add_complete_bipartite(edges, s, a_core)
        for source, host_nodes in zip(s, (b[:2], b[2:4], b[4:6]), strict=True):
            for host_node in host_nodes:
                _add_edge(edges, source, host_node)
    elif mode == "symmetric_boundary":
        a_core = a[:3]
        b_core = b[:3]
        _add_complete_bipartite(edges, s, a_core)
        _add_complete_bipartite(edges, s, b_core)
    elif mode in {"diffuse_no_core_control", "diffuse_no_core_control_v1_1"}:
        a_core = ()
        b_core = ()
        for source, host_nodes in zip(s, (a[:2], a[2:4], a[4:6]), strict=True):
            for host_node in host_nodes:
                _add_edge(edges, source, host_node)
        for source, host_nodes in zip(s, (b[1:3], b[3:5], (b[0], b[5])), strict=True):
            for host_node in host_nodes:
                _add_edge(edges, source, host_node)
        if mode == "diffuse_no_core_control_v1_1":
            for source, host_nodes in zip(s, ((a[0], a[2]), (a[1], a[4]), (a[3], a[5])), strict=True):
                for host_node in host_nodes:
                    _add_edge(edges, source, host_node, 0.5)
            for source, host_nodes in zip(s, ((b[0], b[2]), (b[1], b[4]), (b[3], b[5])), strict=True):
                for host_node in host_nodes:
                    _add_edge(edges, source, host_node, 0.5)
    else:
        raise ValueError(f"unknown absorption mode: {mode}")
    roles = [
        _role(f"{variant_id}:host_a", a, "host", family, "host_a"),
        _role(f"{variant_id}:host_b", b, "host", family, "host_b"),
        _role(f"{variant_id}:small_module", s, "small_module", family, "small_module"),
        _role(f"{variant_id}:host_a_core", a_core, "boundary_core", family, "host_a"),
        _role(f"{variant_id}:host_a_tail", tuple(node for node in a if node not in set(a_core)), "host_tail", family, "host_a"),
        _role(f"{variant_id}:host_b_core", b_core, "boundary_core", family, "host_b"),
        _role(f"{variant_id}:host_b_tail", tuple(node for node in b if node not in set(b_core)), "host_tail", family, "host_b"),
    ]
    if mode == "diffuse_no_core_control_v1_1":
        roles.extend(
            [
                _role(
                    f"{variant_id}:host_a_boundary_core_decoy",
                    (a[0], a[2], a[4]),
                    "boundary_core_decoy",
                    family,
                    "host_a",
                ),
                _role(
                    f"{variant_id}:host_b_boundary_core_decoy",
                    (b[1], b[3], b[5]),
                    "boundary_core_decoy",
                    family,
                    "host_b",
                ),
            ]
        )
    return _graph_from_parts(names, edges), tuple(roles)


def _balanced_graph(
    variant_id: str,
    *,
    mode: str,
) -> tuple[ig.Graph, tuple[RoleAnnotation, ...]]:
    family = "balanced_split"
    a = _nodes(f"{variant_id}_a", 5)
    b = _nodes(f"{variant_id}_b", 5)
    m = _nodes(f"{variant_id}_m", 4)
    names = (*a, *b, *m)
    edges: dict[tuple[str, str], float] = defaultdict(float)
    _add_clique(edges, list(a))
    _add_clique(edges, list(b))
    _add_clique(edges, list(m))
    if mode == "equal_pull":
        a_core = a[:2]
        b_core = b[:2]
        m_a = m[:2]
        m_b = m[2:]
        _add_complete_bipartite(edges, m, a_core)
        _add_complete_bipartite(edges, m, b_core)
        _add_complete_bipartite(edges, m_a, a[2:4])
        _add_complete_bipartite(edges, m_b, b[2:4])
    elif mode == "light_asymmetry":
        a_core = a[:3]
        b_core = b[:2]
        m_a = m[:3]
        m_b = m[3:]
        _add_complete_bipartite(edges, m, a_core)
        _add_complete_bipartite(edges, m, b_core)
        _add_complete_bipartite(edges, m_a, a[3:5])
        _add_complete_bipartite(edges, m_b, b[2:4])
    elif mode == "single_host_dominant_control":
        a_core = a[:3]
        b_core = ()
        m_a = m
        m_b = ()
        _add_complete_bipartite(edges, m, a_core)
        _add_complete_bipartite(edges, m, a[3:5])
        for source, host_node in zip(m, b[:4], strict=True):
            _add_edge(edges, source, host_node, 0.25)
    else:
        raise ValueError(f"unknown balanced mode: {mode}")
    roles = (
        _role(f"{variant_id}:host_a", a, "host", family, "host_a"),
        _role(f"{variant_id}:host_b", b, "host", family, "host_b"),
        _role(f"{variant_id}:middle_module", m, "middle_module", family, "middle_module"),
        _role(f"{variant_id}:middle_pull_a", m_a, "middle_submodule", family, "middle_pull_a"),
        _role(f"{variant_id}:middle_pull_b", m_b, "middle_submodule", family, "middle_pull_b"),
        _role(f"{variant_id}:host_a_core", a_core, "boundary_core", family, "host_a"),
        _role(f"{variant_id}:host_a_tail", tuple(node for node in a if node not in set(a_core)), "host_tail", family, "host_a"),
        _role(f"{variant_id}:host_b_core", b_core, "boundary_core", family, "host_b"),
        _role(f"{variant_id}:host_b_tail", tuple(node for node in b if node not in set(b_core)), "host_tail", family, "host_b"),
    )
    return _graph_from_parts(names, edges), roles


def _diffuse_graph(
    variant_id: str,
    *,
    mode: str,
) -> tuple[ig.Graph, tuple[RoleAnnotation, ...]]:
    family = "diffuse_fragment"
    hosts = {
        host: _nodes(f"{variant_id}_h{host}_", 4)
        for host in range(4)
    }
    weak = _nodes(f"{variant_id}_x", 4)
    names = (*hosts[0], *hosts[1], *hosts[2], *hosts[3], *weak)
    edges: dict[tuple[str, str], float] = defaultdict(float)
    for host_nodes in hosts.values():
        _add_clique(edges, list(host_nodes))
    _add_clique(
        edges,
        list(weak),
        1.5 if mode in {"weak_module_separate_control", "weak_module_separate_control_v1_1"} else 0.5,
    )

    if mode == "one_pair":
        pair_roles = ((weak[0], weak[1]),)
        host_cores = {0: hosts[0][:3], 1: (), 2: hosts[2][:3], 3: ()}
        weak_pair_decoys: tuple[tuple[str, ...], ...] = ()
        host_decoy_cores: dict[int, tuple[str, ...]] = {host: () for host in hosts}
        _add_complete_bipartite(edges, pair_roles[0], host_cores[0])
        _add_complete_bipartite(edges, pair_roles[0], host_cores[2])
        for source, host_node in zip(weak[2:], (hosts[1][0], hosts[3][0]), strict=True):
            _add_edge(edges, source, host_node, 0.5)
    elif mode == "two_pair":
        pair_roles = ((weak[0], weak[1]), (weak[2], weak[3]))
        host_cores = {host: hosts[host][:3] for host in hosts}
        weak_pair_decoys = ()
        host_decoy_cores = {host: () for host in hosts}
        _add_complete_bipartite(edges, pair_roles[0], host_cores[0])
        _add_complete_bipartite(edges, pair_roles[0], host_cores[2])
        _add_complete_bipartite(edges, pair_roles[1], host_cores[1])
        _add_complete_bipartite(edges, pair_roles[1], host_cores[3])
    elif mode in {"weak_module_separate_control", "weak_module_separate_control_v1_1"}:
        pair_roles = ()
        host_cores = {host: () for host in hosts}
        for host, source in zip(hosts, weak, strict=True):
            _add_edge(edges, source, hosts[host][0], 0.25)
        if mode == "weak_module_separate_control_v1_1":
            weak_pair_decoys = ((weak[0], weak[1]), (weak[2], weak[3]))
            host_decoy_cores = {host: hosts[host][:3] for host in hosts}
            _add_complete_bipartite(edges, weak_pair_decoys[0], host_decoy_cores[0], 0.25)
            _add_complete_bipartite(edges, weak_pair_decoys[0], host_decoy_cores[2], 0.25)
            _add_complete_bipartite(edges, weak_pair_decoys[1], host_decoy_cores[1], 0.25)
            _add_complete_bipartite(edges, weak_pair_decoys[1], host_decoy_cores[3], 0.25)
        else:
            weak_pair_decoys = ()
            host_decoy_cores = {host: () for host in hosts}
    else:
        raise ValueError(f"unknown diffuse mode: {mode}")

    role_rows: list[RoleAnnotation] = [
        _role(f"{variant_id}:weak_module", weak, "weak_module", family, "weak_module")
    ]
    for host, host_nodes in hosts.items():
        core = tuple(host_cores[host])
        role_rows.extend(
            [
                _role(f"{variant_id}:host_{host}", host_nodes, "host", family, f"host_{host}"),
                _role(f"{variant_id}:host_{host}_core", core, "boundary_core", family, f"host_{host}"),
                _role(
                    f"{variant_id}:host_{host}_tail",
                    tuple(node for node in host_nodes if node not in set(core)),
                    "host_tail",
                    family,
                    f"host_{host}",
                ),
            ]
        )
    for index, pair_nodes in enumerate(pair_roles):
        role_rows.append(
            _role(f"{variant_id}:weak_pair_{index}", tuple(pair_nodes), "weak_pair", family, f"weak_pair_{index}")
        )
    for host, decoy_core in host_decoy_cores.items():
        if decoy_core:
            role_rows.append(
                _role(
                    f"{variant_id}:host_{host}_boundary_core_decoy",
                    tuple(decoy_core),
                    "boundary_core_decoy",
                    family,
                    f"host_{host}",
                )
            )
    for index, pair_nodes in enumerate(weak_pair_decoys):
        role_rows.append(
            _role(
                f"{variant_id}:weak_pair_decoy_{index}",
                tuple(pair_nodes),
                "weak_pair_decoy",
                family,
                f"weak_pair_decoy_{index}",
            )
        )
    return _graph_from_parts(names, edges), tuple(role_rows)


def _variant_cases(seed_count: int, *, panel_version: str = "v1") -> list[VariantGraphCase]:
    if panel_version not in {"v1", "v1_1", "v1_2"}:
        raise ValueError(f"unknown panel version: {panel_version}")
    uses_v1_1_controls = panel_version in {"v1_1", "v1_2"}
    absorption_control_mode = (
        "diffuse_no_core_control_v1_1" if uses_v1_1_controls else "diffuse_no_core_control"
    )
    diffuse_control_mode = (
        "weak_module_separate_control_v1_1" if uses_v1_1_controls else "weak_module_separate_control"
    )
    cases = [
        VariantGraphCase(
            variant_id="nt_symmetric_tie_anchor",
            mechanism_family="near_tie_bridge",
            mechanism_state="preserved_anchor",
            gamma=0.50,
            seed_count=seed_count,
            expected_baseline_behavior="bridge joins either host with near-equal quality",
            responsible_rule="bridge-contact",
            control_read="calibration anchor, not generalization credit",
            builder=lambda: _near_tie_graph("nt_symmetric_tie_anchor", a_contact=4, b_contact=4),
        ),
        VariantGraphCase(
            variant_id="nt_light_bias_a",
            mechanism_family="near_tie_bridge",
            mechanism_state="perturbed_preserved",
            gamma=0.50,
            seed_count=seed_count,
            expected_baseline_behavior="host A has a mild contact advantage while host B remains plausible",
            responsible_rule="bridge-contact",
            control_read="success means calibrated dominance, not forced symmetry",
            builder=lambda: _near_tie_graph("nt_light_bias_a", a_contact=5, b_contact=4),
        ),
        VariantGraphCase(
            variant_id="nt_hard_bias_a_control",
            mechanism_family="near_tie_bridge",
            mechanism_state="mechanism_removed_control",
            gamma=0.50,
            seed_count=seed_count,
            expected_baseline_behavior="one dominant bridge assignment should remain",
            responsible_rule="bridge-contact",
            control_read="failing to recover side B is correct",
            builder=lambda: _near_tie_graph("nt_hard_bias_a_control", a_contact=5, b_contact=1, b_weight=0.5),
        ),
        VariantGraphCase(
            variant_id="ab_boundary_vs_diffuse",
            mechanism_family="absorption_triad",
            mechanism_state="preserved",
            gamma=0.75,
            seed_count=seed_count,
            expected_baseline_behavior="compact boundary-core host competes with diffuse host contact",
            responsible_rule="boundary-core",
            control_read="target hard endpoint should be reachable if recurrent",
            builder=lambda: _absorption_graph("ab_boundary_vs_diffuse", mode="boundary_vs_diffuse"),
        ),
        VariantGraphCase(
            variant_id="ab_symmetric_boundary",
            mechanism_family="absorption_triad",
            mechanism_state="preserved_ambiguous",
            gamma=0.75,
            seed_count=seed_count,
            expected_baseline_behavior="two compact host boundary cores compete",
            responsible_rule="boundary-core",
            control_read="both host absorptions should be reachable if recurrent",
            builder=lambda: _absorption_graph("ab_symmetric_boundary", mode="symmetric_boundary"),
        ),
        VariantGraphCase(
            variant_id="ab_diffuse_no_core_control",
            mechanism_family="absorption_triad",
            mechanism_state="mechanism_removed_control",
            gamma=0.75,
            seed_count=seed_count,
            expected_baseline_behavior="absorption, if any, lacks a compact boundary core",
            responsible_rule="boundary-core",
            control_read="boundary-core rule should not explain a hard endpoint",
            builder=lambda: _absorption_graph("ab_diffuse_no_core_control", mode=absorption_control_mode),
        ),
        VariantGraphCase(
            variant_id="bs_equal_pull",
            mechanism_family="balanced_split",
            mechanism_state="preserved",
            gamma=0.65,
            seed_count=seed_count,
            expected_baseline_behavior="middle module has balanced host pull",
            responsible_rule="boundary-core/contact-split",
            control_read="split and collapse alternatives should recur",
            builder=lambda: _balanced_graph("bs_equal_pull", mode="equal_pull"),
        ),
        VariantGraphCase(
            variant_id="bs_light_asymmetry",
            mechanism_family="balanced_split",
            mechanism_state="perturbed_preserved",
            gamma=0.65,
            seed_count=seed_count,
            expected_baseline_behavior="one host is favored but alternatives remain possible",
            responsible_rule="boundary-core",
            control_read="success may be lower-rate but mechanism-readable",
            builder=lambda: _balanced_graph("bs_light_asymmetry", mode="light_asymmetry"),
        ),
        VariantGraphCase(
            variant_id="bs_single_host_dominant_control",
            mechanism_family="balanced_split",
            mechanism_state="mechanism_removed_control",
            gamma=0.65,
            seed_count=seed_count,
            expected_baseline_behavior="middle module should collapse to one host",
            responsible_rule="boundary-core",
            control_read="missing the opposite collapse is correct",
            builder=lambda: _balanced_graph("bs_single_host_dominant_control", mode="single_host_dominant_control"),
        ),
        VariantGraphCase(
            variant_id="df_one_pair",
            mechanism_family="diffuse_fragment",
            mechanism_state="preserved_minimal",
            gamma=0.65,
            seed_count=seed_count,
            expected_baseline_behavior="one weak-node pair creates a compact diffuse alternative",
            responsible_rule="weak-pair tail-split",
            control_read="target pair endpoint should be reachable if recurrent",
            builder=lambda: _diffuse_graph("df_one_pair", mode="one_pair"),
        ),
        VariantGraphCase(
            variant_id="df_two_pair",
            mechanism_family="diffuse_fragment",
            mechanism_state="preserved_robust",
            gamma=0.65,
            seed_count=seed_count,
            expected_baseline_behavior="two weak-node pairs create multiple diffuse alternatives",
            responsible_rule="weak-pair tail-split",
            control_read="stronger version of the Stress 3 mechanism",
            builder=lambda: _diffuse_graph("df_two_pair", mode="two_pair"),
        ),
        VariantGraphCase(
            variant_id="df_weak_module_separate_control",
            mechanism_family="diffuse_fragment",
            mechanism_state="mechanism_removed_control",
            gamma=0.65,
            seed_count=seed_count,
            expected_baseline_behavior="weak module remains coherent or separate",
            responsible_rule="weak-pair tail-split",
            control_read="weak-pair rule should not create a false diffuse claim",
            builder=lambda: _diffuse_graph("df_weak_module_separate_control", mode=diffuse_control_mode),
        ),
    ]
    return [replace(case, panel_version=panel_version) for case in cases]


def _edge_rows(graph: ig.Graph, variant_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    has_weight = "weight" in graph.es.attributes()
    for edge in graph.es:
        left = str(graph.vs[edge.source]["name"])
        right = str(graph.vs[edge.target]["name"])
        rows.append(
            {
                "variant_id": variant_id,
                "left_node": min(left, right),
                "right_node": max(left, right),
                "weight": float(edge["weight"]) if has_weight else 1.0,
            }
        )
    return sorted(rows, key=lambda row: (row["variant_id"], row["left_node"], row["right_node"]))


def _edge_weight_lookup(graph: ig.Graph) -> dict[tuple[str, str], float]:
    weights: dict[tuple[str, str], float] = {}
    has_weight = "weight" in graph.es.attributes()
    for edge in graph.es:
        left = str(graph.vs[edge.source]["name"])
        right = str(graph.vs[edge.target]["name"])
        weight = float(edge["weight"]) if has_weight else 1.0
        weights[tuple(sorted((left, right)))] = weight
    return weights


def _edge_weight(weights: dict[tuple[str, str], float], left: str, right: str) -> float:
    return weights.get(tuple(sorted((left, right))), 0.0)


def _contact_weight(
    weights: dict[tuple[str, str], float],
    left_nodes: tuple[str, ...],
    right_nodes: tuple[str, ...],
) -> float:
    return float(
        sum(
            _edge_weight(weights, left, right)
            for left in left_nodes
            for right in right_nodes
        )
    )


def _graph_hash(graph: ig.Graph) -> str:
    payload = {
        "nodes": sorted(map(str, graph.vs["name"])),
        "edges": _edge_rows(graph, "graph"),
    }
    return _sha256_json(payload)


def _roles_hash(roles: tuple[RoleAnnotation, ...]) -> str:
    payload = [
        {
            "role_id": role.role_id,
            "node_ids": sorted(role.node_ids),
            "role_type": role.role_type,
            "mechanism_family": role.mechanism_family,
            "role_slot": role.role_slot,
            "allowed": role.allowed_for_candidate_generation,
        }
        for role in sorted(roles, key=lambda item: item.role_id)
    ]
    return _sha256_json(payload)


def _opaque_bundle(
    graph: ig.Graph,
    roles: tuple[RoleAnnotation, ...],
) -> tuple[ig.Graph, tuple[RoleAnnotation, ...], tuple[RoleAnnotation, ...], dict[str, str]]:
    canonical_names = sorted(map(str, graph.vs["name"]))
    node_id_map = {name: f"n{index:04d}" for index, name in enumerate(canonical_names)}
    opaque_edges: dict[tuple[str, str], float] = defaultdict(float)
    for edge in graph.es:
        left = str(graph.vs[edge.source]["name"])
        right = str(graph.vs[edge.target]["name"])
        weight = float(edge["weight"]) if "weight" in graph.es.attributes() else 1.0
        _add_edge(opaque_edges, node_id_map[left], node_id_map[right], weight)
    opaque_graph = _build_graph(list(node_id_map.values()), opaque_edges)
    opaque_roles = tuple(
        RoleAnnotation(
            role_id=role.role_id,
            node_ids=tuple(node_id_map[node] for node in role.node_ids),
            role_type=role.role_type,
            mechanism_family=role.mechanism_family,
            role_slot=role.role_slot,
            allowed_for_candidate_generation=role.allowed_for_candidate_generation,
        )
        for role in roles
    )
    permuted_roles = tuple(
        RoleAnnotation(
            role_id=f"r{index:04d}",
            node_ids=role.node_ids,
            role_type=role.role_type,
            mechanism_family=role.mechanism_family,
            role_slot=role.role_slot,
            allowed_for_candidate_generation=role.allowed_for_candidate_generation,
        )
        for index, role in enumerate(sorted(opaque_roles, key=lambda item: item.role_id))
    )
    return opaque_graph, opaque_roles, permuted_roles, node_id_map


def _materialize_bundles(cases: list[VariantGraphCase]) -> list[GraphBundle]:
    bundles: list[GraphBundle] = []
    for case in cases:
        graph, roles = case.builder()
        opaque_graph, opaque_roles, role_name_permuted_roles, node_id_map = _opaque_bundle(graph, roles)
        bundles.append(
            GraphBundle(
                case=case,
                canonical_graph=graph,
                roles=roles,
                opaque_graph=opaque_graph,
                opaque_roles=opaque_roles,
                role_name_permuted_roles=role_name_permuted_roles,
                node_id_map=node_id_map,
                graph_hash=_graph_hash(graph),
                role_hash=_roles_hash(roles),
            )
        )
    return bundles


def _role_by_type(roles: tuple[RoleAnnotation, ...], role_type: str) -> list[RoleAnnotation]:
    return sorted(
        [role for role in roles if role.role_type == role_type and role.allowed_for_candidate_generation],
        key=lambda role: (role.role_slot, role.role_id),
    )


def _role_for_slot(
    roles: tuple[RoleAnnotation, ...],
    *,
    role_type: str,
    role_slot: str,
) -> RoleAnnotation | None:
    matches = [
        role
        for role in roles
        if role.role_type == role_type and role.role_slot == role_slot and role.allowed_for_candidate_generation
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda role: role.role_id)[0]


def _roles_for_slot(
    roles: tuple[RoleAnnotation, ...],
    *,
    role_type: str,
    role_slot: str,
) -> list[RoleAnnotation]:
    return sorted(
        [
            role
            for role in roles
            if role.role_type == role_type and role.role_slot == role_slot and role.allowed_for_candidate_generation
        ],
        key=lambda role: role.role_id,
    )


def _role_nodes(role: RoleAnnotation | None) -> tuple[str, ...]:
    if role is None:
        return ()
    return tuple(role.node_ids)


def _graph_for_roles(bundle: GraphBundle, roles: tuple[RoleAnnotation, ...]) -> ig.Graph:
    canonical_names = set(map(str, bundle.canonical_graph.vs["name"]))
    role_nodes = {node for role in roles for node in role.node_ids}
    if role_nodes and role_nodes.isdisjoint(canonical_names):
        return bundle.opaque_graph
    return bundle.canonical_graph


def _candidate_signature(groups: tuple[tuple[str, ...], ...]) -> str:
    normalized = sorted(sorted(group) for group in groups)
    return _sha256_json(normalized, length=12)


def _candidate_class_signature(
    *,
    handle_type: str,
    target_mechanism_class: str,
    initial_role_groups: tuple[tuple[str, ...], ...],
    graph_rule: str,
) -> str:
    payload = {
        "handle_type": handle_type,
        "target_mechanism_class": target_mechanism_class,
        "initial_role_groups": sorted(sorted(group) for group in initial_role_groups),
        "graph_rule": graph_rule,
    }
    return _sha256_json(payload, length=12)


def _candidate(
    *,
    variant_id: str,
    candidate_slug: str,
    handle_type: str,
    target_mechanism_class: str,
    target_claim_allowed: bool,
    initial_groups: tuple[tuple[str, ...], ...],
    initial_role_groups: tuple[tuple[str, ...], ...],
    touched_nodes: tuple[str, ...],
    evidence_role_ids: tuple[str, ...],
    graph_rule: str,
    graph_evidence: dict[str, Any],
) -> VariantCandidate:
    class_sig = _candidate_class_signature(
        handle_type=handle_type,
        target_mechanism_class=target_mechanism_class,
        initial_role_groups=initial_role_groups,
        graph_rule=graph_rule,
    )
    return VariantCandidate(
        variant_id=variant_id,
        handle_candidate_id=f"{variant_id}__{candidate_slug}",
        handle_type=handle_type,
        target_mechanism_class=target_mechanism_class,
        target_claim_allowed=target_claim_allowed,
        initial_groups=initial_groups,
        initial_role_groups=initial_role_groups,
        touched_nodes=tuple(sorted(set(touched_nodes))),
        evidence_role_ids=tuple(sorted(evidence_role_ids)),
        graph_rule=graph_rule,
        graph_evidence=graph_evidence,
        candidate_signature=_candidate_signature(initial_groups),
        candidate_class_signature=class_sig,
    )


def _other_host_groups(hosts: list[RoleAnnotation], selected_slot: str) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(host.node_ids) for host in hosts if host.role_slot != selected_slot)


def _bridge_candidates(bundle: GraphBundle, roles: tuple[RoleAnnotation, ...]) -> list[VariantCandidate]:
    bridge = _role_for_slot(roles, role_type="bridge", role_slot="bridge")
    hosts = _role_by_type(roles, "host")
    if bridge is None:
        return []
    weights = _edge_weight_lookup(_graph_for_roles(bundle, roles))
    target_allowed = "control" not in bundle.case.mechanism_state
    rows: list[VariantCandidate] = []
    for host in hosts:
        other_groups = _other_host_groups(hosts, host.role_slot)
        rows.append(
            _candidate(
                variant_id=bundle.case.variant_id,
                candidate_slug=f"bridge_to_{host.role_slot}",
                handle_type="blind_bridge_contact_initialization",
                target_mechanism_class=f"bridge_to_{host.role_slot}",
                target_claim_allowed=target_allowed,
                initial_groups=((*(host.node_ids), *(bridge.node_ids)), *other_groups),
                initial_role_groups=((host.role_slot, bridge.role_slot), *(tuple(other.role_slot for other in hosts if other.role_slot != host.role_slot),)),
                touched_nodes=bridge.node_ids,
                evidence_role_ids=(bridge.role_id, host.role_id),
                graph_rule="attach bridge role to each declared host role using graph contact only",
                graph_evidence={
                    "bridge_role": bridge.role_id,
                    "host_role": host.role_id,
                    "bridge_host_contact_weight": _contact_weight(weights, bridge.node_ids, host.node_ids),
                    "input_policy": "P2 graph manifest, roles, and edge weights only",
                },
            )
        )
    rows.append(
        _candidate(
            variant_id=bundle.case.variant_id,
            candidate_slug="bridge_separate_control",
            handle_type="blind_bridge_control_initialization",
            target_mechanism_class="bridge_separate_control",
            target_claim_allowed=False,
            initial_groups=(*(tuple(host.node_ids) for host in hosts), bridge.node_ids),
            initial_role_groups=(*(tuple([host.role_slot]) for host in hosts), (bridge.role_slot,)),
            touched_nodes=bridge.node_ids,
            evidence_role_ids=(bridge.role_id,),
            graph_rule="keep bridge role separate as a negative-control handle",
            graph_evidence={
                "bridge_role": bridge.role_id,
                "input_policy": "P2 graph manifest, roles, and edge weights only",
            },
        )
    )
    return rows


def _boundary_core_candidates(
    bundle: GraphBundle,
    roles: tuple[RoleAnnotation, ...],
    *,
    module_role_type: str,
    module_slot: str,
    handle_prefix: str,
) -> list[VariantCandidate]:
    module = _role_for_slot(roles, role_type=module_role_type, role_slot=module_slot)
    hosts = _role_by_type(roles, "host")
    if module is None:
        return []
    weights = _edge_weight_lookup(_graph_for_roles(bundle, roles))
    target_allowed = "control" not in bundle.case.mechanism_state
    rows: list[VariantCandidate] = []
    for host in hosts:
        core = _role_for_slot(roles, role_type="boundary_core", role_slot=host.role_slot)
        tail = _role_for_slot(roles, role_type="host_tail", role_slot=host.role_slot)
        for decoy_core in _roles_for_slot(roles, role_type="boundary_core_decoy", role_slot=host.role_slot):
            if not decoy_core.node_ids:
                continue
            other_groups = _other_host_groups(hosts, host.role_slot)
            rows.append(
                _candidate(
                    variant_id=bundle.case.variant_id,
                    candidate_slug=f"{handle_prefix}_to_{host.role_slot}_boundary_core_decoy",
                    handle_type=f"blind_{handle_prefix}_boundary_core_initialization",
                    target_mechanism_class=f"{handle_prefix}_to_{host.role_slot}_boundary_core_decoy_control",
                    target_claim_allowed=False,
                    initial_groups=((*(decoy_core.node_ids), *(module.node_ids)), _role_nodes(tail), *other_groups),
                    initial_role_groups=(
                        (f"{decoy_core.role_slot}_boundary_core_decoy", module.role_slot),
                        (f"{tail.role_slot}_tail" if tail else "missing_tail",),
                        *(tuple([other.role_slot]) for other in hosts if other.role_slot != host.role_slot),
                    ),
                    touched_nodes=(*(module.node_ids), *(decoy_core.node_ids)),
                    evidence_role_ids=(module.role_id, host.role_id, decoy_core.role_id, tail.role_id if tail else ""),
                    graph_rule="attach ambiguous module to target-like decoy boundary-core role without positive claim",
                    graph_evidence={
                        "module_role": module.role_id,
                        "host_role": host.role_id,
                        "boundary_core_decoy_role": decoy_core.role_id,
                        "tail_role": tail.role_id if tail else None,
                        "module_host_contact_weight": _contact_weight(weights, module.node_ids, host.node_ids),
                        "module_decoy_core_contact_weight": _contact_weight(weights, module.node_ids, decoy_core.node_ids),
                        "boundary_core_decoy_node_count": len(decoy_core.node_ids),
                        "target_claim_allowed": False,
                        "input_policy": "P2 graph manifest, roles, and edge weights only",
                    },
                )
            )
        if core is None or not core.node_ids:
            continue
        other_groups = _other_host_groups(hosts, host.role_slot)
        rows.append(
            _candidate(
                variant_id=bundle.case.variant_id,
                candidate_slug=f"{handle_prefix}_to_{host.role_slot}_boundary_core",
                handle_type=f"blind_{handle_prefix}_boundary_core_initialization",
                target_mechanism_class=f"{handle_prefix}_to_{host.role_slot}_boundary_core",
                target_claim_allowed=target_allowed,
                initial_groups=((*(core.node_ids), *(module.node_ids)), _role_nodes(tail), *other_groups),
                initial_role_groups=((core.role_slot, module.role_slot), (f"{tail.role_slot}_tail",), *(tuple([other.role_slot]) for other in hosts if other.role_slot != host.role_slot)),
                touched_nodes=(*(module.node_ids), *(core.node_ids)),
                evidence_role_ids=(module.role_id, host.role_id, core.role_id, tail.role_id if tail else ""),
                graph_rule="attach ambiguous module to declared host boundary-core role",
                graph_evidence={
                    "module_role": module.role_id,
                    "host_role": host.role_id,
                    "boundary_core_role": core.role_id,
                    "tail_role": tail.role_id if tail else None,
                    "module_host_contact_weight": _contact_weight(weights, module.node_ids, host.node_ids),
                    "module_core_contact_weight": _contact_weight(weights, module.node_ids, core.node_ids),
                    "boundary_core_node_count": len(core.node_ids),
                    "input_policy": "P2 graph manifest, roles, and edge weights only",
                },
            )
        )
    rows.append(
        _candidate(
            variant_id=bundle.case.variant_id,
            candidate_slug=f"{handle_prefix}_separate_control",
            handle_type=f"blind_{handle_prefix}_control_initialization",
            target_mechanism_class=f"{handle_prefix}_separate_control",
            target_claim_allowed=False,
            initial_groups=(*(tuple(host.node_ids) for host in hosts), module.node_ids),
            initial_role_groups=(*(tuple([host.role_slot]) for host in hosts), (module.role_slot,)),
            touched_nodes=module.node_ids,
            evidence_role_ids=(module.role_id,),
            graph_rule="keep ambiguous module separate as a negative-control handle",
            graph_evidence={
                "module_role": module.role_id,
                "input_policy": "P2 graph manifest, roles, and edge weights only",
            },
        )
    )
    return rows


def _balanced_contact_split_candidates(
    bundle: GraphBundle,
    roles: tuple[RoleAnnotation, ...],
) -> list[VariantCandidate]:
    middle_a = _role_for_slot(roles, role_type="middle_submodule", role_slot="middle_pull_a")
    middle_b = _role_for_slot(roles, role_type="middle_submodule", role_slot="middle_pull_b")
    host_a = _role_for_slot(roles, role_type="host", role_slot="host_a")
    host_b = _role_for_slot(roles, role_type="host", role_slot="host_b")
    if middle_a is None or middle_b is None or host_a is None or host_b is None:
        return []
    if not middle_a.node_ids or not middle_b.node_ids:
        return []
    weights = _edge_weight_lookup(_graph_for_roles(bundle, roles))
    target_allowed = "control" not in bundle.case.mechanism_state
    return [
        _candidate(
            variant_id=bundle.case.variant_id,
            candidate_slug="middle_contact_split",
            handle_type="blind_middle_contact_split_initialization",
            target_mechanism_class="middle_contact_split",
            target_claim_allowed=target_allowed,
            initial_groups=((*(host_a.node_ids), *(middle_a.node_ids)), (*(host_b.node_ids), *(middle_b.node_ids))),
            initial_role_groups=((host_a.role_slot, middle_a.role_slot), (host_b.role_slot, middle_b.role_slot)),
            touched_nodes=(*(middle_a.node_ids), *(middle_b.node_ids)),
            evidence_role_ids=(host_a.role_id, host_b.role_id, middle_a.role_id, middle_b.role_id),
            graph_rule="split middle submodules by declared host-contact pull roles",
            graph_evidence={
                "middle_pull_a_role": middle_a.role_id,
                "middle_pull_b_role": middle_b.role_id,
                "pull_a_contact_to_host_a": _contact_weight(weights, middle_a.node_ids, host_a.node_ids),
                "pull_a_contact_to_host_b": _contact_weight(weights, middle_a.node_ids, host_b.node_ids),
                "pull_b_contact_to_host_a": _contact_weight(weights, middle_b.node_ids, host_a.node_ids),
                "pull_b_contact_to_host_b": _contact_weight(weights, middle_b.node_ids, host_b.node_ids),
                "input_policy": "P2 graph manifest, roles, and edge weights only",
            },
        )
    ]


def _joint_weak_pair_candidates(
    bundle: GraphBundle,
    roles: tuple[RoleAnnotation, ...],
    *,
    weak_module: RoleAnnotation,
    hosts: list[RoleAnnotation],
    pair_roles: list[RoleAnnotation],
    core_role_type: str,
    target_mechanism_class: str,
    target_claim_allowed: bool,
    decoy: bool,
) -> list[VariantCandidate]:
    if len(pair_roles) < 2:
        return []
    weights = _edge_weight_lookup(_graph_for_roles(bundle, roles))
    components: list[dict[str, Any]] = []
    for pair in pair_roles:
        for host in hosts:
            core = _role_for_slot(roles, role_type=core_role_type, role_slot=host.role_slot)
            tail = _role_for_slot(roles, role_type="host_tail", role_slot=host.role_slot)
            if core is None or not core.node_ids:
                continue
            pair_core_contact = _contact_weight(weights, pair.node_ids, core.node_ids)
            if pair_core_contact <= 0:
                continue
            components.append(
                {
                    "pair": pair,
                    "host": host,
                    "core": core,
                    "tail": tail,
                    "pair_core_contact": pair_core_contact,
                    "pair_host_contact": _contact_weight(weights, pair.node_ids, host.node_ids),
                }
            )

    rows: list[VariantCandidate] = []
    for combo in combinations(components, 2):
        if len({str(item["pair"].role_slot) for item in combo}) != len(combo):
            continue
        if len({str(item["host"].role_slot) for item in combo}) != len(combo):
            continue
        ordered = sorted(combo, key=lambda item: (str(item["pair"].role_slot), str(item["host"].role_slot)))
        selected_pair_nodes = {
            node
            for item in ordered
            for node in item["pair"].node_ids
        }
        selected_host_slots = {str(item["host"].role_slot) for item in ordered}
        initial_groups: list[tuple[str, ...]] = []
        initial_role_groups: list[tuple[str, ...]] = []
        graph_components: list[dict[str, Any]] = []
        evidence_role_ids: list[str] = [weak_module.role_id]
        touched_nodes: list[str] = []
        slug_parts: list[str] = []
        for item in ordered:
            pair = item["pair"]
            host = item["host"]
            core = item["core"]
            tail = item["tail"]
            initial_groups.append((*(core.node_ids), *(pair.node_ids)))
            core_role_label = f"{core.role_slot}_boundary_core_decoy" if decoy else str(core.role_slot)
            initial_role_groups.append((core_role_label, str(pair.role_slot)))
            tail_nodes = _role_nodes(tail)
            if tail_nodes:
                initial_groups.append(tail_nodes)
                initial_role_groups.append((f"{tail.role_slot}_tail",))
            evidence_role_ids.extend([pair.role_id, host.role_id, core.role_id])
            if tail is not None:
                evidence_role_ids.append(tail.role_id)
            touched_nodes.extend([*pair.node_ids, *core.node_ids])
            slug_parts.append(f"{pair.role_slot}_to_{host.role_slot}")
            graph_components.append(
                {
                    "weak_pair_role": pair.role_id,
                    "host_role": host.role_id,
                    "core_role": core.role_id,
                    "tail_role": tail.role_id if tail else None,
                    "pair_core_contact_weight": float(item["pair_core_contact"]),
                    "pair_host_contact_weight": float(item["pair_host_contact"]),
                }
            )

        residual_weak = tuple(node for node in weak_module.node_ids if node not in selected_pair_nodes)
        if residual_weak:
            initial_groups.append(residual_weak)
            initial_role_groups.append(("weak_module_residual",))
        for host in hosts:
            if host.role_slot in selected_host_slots:
                continue
            initial_groups.append(tuple(host.node_ids))
            initial_role_groups.append((host.role_slot,))

        rows.append(
            _candidate(
                variant_id=bundle.case.variant_id,
                candidate_slug=f"joint_{'__'.join(slug_parts)}_tail_split{'_decoy' if decoy else ''}",
                handle_type="blind_joint_weak_pair_tail_split_initialization",
                target_mechanism_class=target_mechanism_class,
                target_claim_allowed=target_claim_allowed,
                initial_groups=tuple(initial_groups),
                initial_role_groups=tuple(initial_role_groups),
                touched_nodes=tuple(touched_nodes),
                evidence_role_ids=tuple(evidence_role_ids),
                graph_rule=(
                    "jointly attach multiple declared weak-pair roles to contacted host boundary cores "
                    "and split each selected host tail"
                    if not decoy
                    else "jointly attach weak-pair-like decoy roles to contacted host decoy cores without positive claim"
                ),
                graph_evidence={
                    "weak_module_role": weak_module.role_id,
                    "joint_component_count": len(ordered),
                    "joint_components": graph_components,
                    "target_claim_allowed": bool(target_claim_allowed),
                    "input_policy": "P2 graph manifest, roles, and edge weights only",
                },
            )
        )
    return rows


def _weak_pair_candidates(bundle: GraphBundle, roles: tuple[RoleAnnotation, ...]) -> list[VariantCandidate]:
    weak_module = _role_for_slot(roles, role_type="weak_module", role_slot="weak_module")
    hosts = _role_by_type(roles, "host")
    weak_pairs = _role_by_type(roles, "weak_pair")
    weak_pair_decoys = _role_by_type(roles, "weak_pair_decoy")
    if weak_module is None:
        return []
    weights = _edge_weight_lookup(_graph_for_roles(bundle, roles))
    target_allowed = "control" not in bundle.case.mechanism_state
    rows: list[VariantCandidate] = []
    for pair in weak_pairs:
        for host in hosts:
            core = _role_for_slot(roles, role_type="boundary_core", role_slot=host.role_slot)
            tail = _role_for_slot(roles, role_type="host_tail", role_slot=host.role_slot)
            if core is None or not core.node_ids:
                continue
            pair_core_contact = _contact_weight(weights, pair.node_ids, core.node_ids)
            if pair_core_contact <= 0:
                continue
            other_hosts = _other_host_groups(hosts, host.role_slot)
            residual_weak = tuple(node for node in weak_module.node_ids if node not in set(pair.node_ids))
            rows.append(
                _candidate(
                    variant_id=bundle.case.variant_id,
                    candidate_slug=f"{pair.role_slot}_to_{host.role_slot}_tail_split",
                    handle_type="blind_weak_pair_tail_split_initialization",
                    target_mechanism_class="weak_pair_tail_split",
                    target_claim_allowed=target_allowed,
                    initial_groups=((*(core.node_ids), *(pair.node_ids)), _role_nodes(tail), residual_weak, *other_hosts),
                    initial_role_groups=((core.role_slot, pair.role_slot), (f"{tail.role_slot}_tail",), ("weak_module_residual",), *(tuple([other.role_slot]) for other in hosts if other.role_slot != host.role_slot)),
                    touched_nodes=(*(pair.node_ids), *(core.node_ids)),
                    evidence_role_ids=(weak_module.role_id, pair.role_id, host.role_id, core.role_id, tail.role_id if tail else ""),
                    graph_rule="attach declared weak-pair role to contacted host boundary core and split host tail",
                    graph_evidence={
                        "weak_module_role": weak_module.role_id,
                        "weak_pair_role": pair.role_id,
                        "host_role": host.role_id,
                        "boundary_core_role": core.role_id,
                        "weak_pair_core_contact_weight": pair_core_contact,
                        "weak_pair_host_contact_weight": _contact_weight(weights, pair.node_ids, host.node_ids),
                        "input_policy": "P2 graph manifest, roles, and edge weights only",
                    },
                )
            )
    for pair in weak_pair_decoys:
        for host in hosts:
            decoy_core = _role_for_slot(roles, role_type="boundary_core_decoy", role_slot=host.role_slot)
            tail = _role_for_slot(roles, role_type="host_tail", role_slot=host.role_slot)
            if decoy_core is None or not decoy_core.node_ids:
                continue
            pair_core_contact = _contact_weight(weights, pair.node_ids, decoy_core.node_ids)
            if pair_core_contact <= 0:
                continue
            other_hosts = _other_host_groups(hosts, host.role_slot)
            residual_weak = tuple(node for node in weak_module.node_ids if node not in set(pair.node_ids))
            rows.append(
                _candidate(
                    variant_id=bundle.case.variant_id,
                    candidate_slug=f"{pair.role_slot}_to_{host.role_slot}_tail_split_decoy",
                    handle_type="blind_weak_pair_tail_split_initialization",
                    target_mechanism_class="weak_pair_tail_split_decoy_control",
                    target_claim_allowed=False,
                    initial_groups=(
                        (*(decoy_core.node_ids), *(pair.node_ids)),
                        _role_nodes(tail),
                        residual_weak,
                        *other_hosts,
                    ),
                    initial_role_groups=(
                        (f"{decoy_core.role_slot}_boundary_core_decoy", pair.role_slot),
                        (f"{tail.role_slot}_tail",),
                        ("weak_module_residual",),
                        *(tuple([other.role_slot]) for other in hosts if other.role_slot != host.role_slot),
                    ),
                    touched_nodes=(*(pair.node_ids), *(decoy_core.node_ids)),
                    evidence_role_ids=(
                        weak_module.role_id,
                        pair.role_id,
                        host.role_id,
                        decoy_core.role_id,
                        tail.role_id if tail else "",
                    ),
                    graph_rule="attach weak-pair-like decoy role to contacted host decoy core without positive claim",
                    graph_evidence={
                        "weak_module_role": weak_module.role_id,
                        "weak_pair_decoy_role": pair.role_id,
                        "host_role": host.role_id,
                        "boundary_core_decoy_role": decoy_core.role_id,
                        "weak_pair_decoy_core_contact_weight": pair_core_contact,
                        "weak_pair_decoy_host_contact_weight": _contact_weight(weights, pair.node_ids, host.node_ids),
                        "target_claim_allowed": False,
                        "input_policy": "P2 graph manifest, roles, and edge weights only",
                    },
                )
            )
    if bundle.case.panel_version == "v1_2":
        rows.extend(
            _joint_weak_pair_candidates(
                bundle,
                roles,
                weak_module=weak_module,
                hosts=hosts,
                pair_roles=weak_pairs,
                core_role_type="boundary_core",
                target_mechanism_class="joint_weak_pair_tail_split",
                target_claim_allowed=target_allowed,
                decoy=False,
            )
        )
        rows.extend(
            _joint_weak_pair_candidates(
                bundle,
                roles,
                weak_module=weak_module,
                hosts=hosts,
                pair_roles=weak_pair_decoys,
                core_role_type="boundary_core_decoy",
                target_mechanism_class="joint_weak_pair_tail_split_decoy_control",
                target_claim_allowed=False,
                decoy=True,
            )
        )
    rows.append(
        _candidate(
            variant_id=bundle.case.variant_id,
            candidate_slug="weak_module_separate_control",
            handle_type="blind_weak_module_control_initialization",
            target_mechanism_class="weak_module_separate_control",
            target_claim_allowed=False,
            initial_groups=(*(tuple(host.node_ids) for host in hosts), weak_module.node_ids),
            initial_role_groups=(*(tuple([host.role_slot]) for host in hosts), (weak_module.role_slot,)),
            touched_nodes=weak_module.node_ids,
            evidence_role_ids=(weak_module.role_id,),
            graph_rule="keep weak module separate as a negative-control handle",
            graph_evidence={
                "weak_module_role": weak_module.role_id,
                "input_policy": "P2 graph manifest, roles, and edge weights only",
            },
        )
    )
    return rows


def _generate_candidates(bundle: GraphBundle, roles: tuple[RoleAnnotation, ...] | None = None) -> list[VariantCandidate]:
    active_roles = bundle.roles if roles is None else roles
    if bundle.case.mechanism_family == "near_tie_bridge":
        return _bridge_candidates(bundle, active_roles)
    if bundle.case.mechanism_family == "absorption_triad":
        return _boundary_core_candidates(
            bundle,
            active_roles,
            module_role_type="small_module",
            module_slot="small_module",
            handle_prefix="small_module",
        )
    if bundle.case.mechanism_family == "balanced_split":
        return [
            *_balanced_contact_split_candidates(bundle, active_roles),
            *_boundary_core_candidates(
                bundle,
                active_roles,
                module_role_type="middle_module",
                module_slot="middle_module",
                handle_prefix="middle_module",
            ),
        ]
    if bundle.case.mechanism_family == "diffuse_fragment":
        return _weak_pair_candidates(bundle, active_roles)
    raise ValueError(f"unknown mechanism family: {bundle.case.mechanism_family}")


def _manifest_frame(bundles: list[GraphBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        graph = bundle.canonical_graph
        total_weight = sum(float(weight) for weight in graph.es["weight"]) if graph.ecount() else 0.0
        rows.append(
            {
                "variant_id": bundle.case.variant_id,
                "mechanism_family": bundle.case.mechanism_family,
                "mechanism_state": bundle.case.mechanism_state,
                "gamma": bundle.case.gamma,
                "seed_count": bundle.case.seed_count,
                "node_count": graph.vcount(),
                "edge_count": graph.ecount(),
                "total_edge_weight": total_weight,
                "graph_hash": bundle.graph_hash,
                "role_hash": bundle.role_hash,
                "expected_baseline_behavior": bundle.case.expected_baseline_behavior,
                "responsible_rule": bundle.case.responsible_rule,
                "control_read": bundle.case.control_read,
                "construction_phase": "P0_fixed_manifest",
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values("variant_id"))


def _edges_frame(bundles: list[GraphBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        rows.extend(_edge_rows(bundle.canonical_graph, bundle.case.variant_id))
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "left_node", "right_node"]))


def _roles_frame(bundles: list[GraphBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        for role in sorted(bundle.roles, key=lambda item: item.role_id):
            rows.append(
                {
                    "variant_id": bundle.case.variant_id,
                    "role_id": role.role_id,
                    "role_type": role.role_type,
                    "role_slot": role.role_slot,
                    "mechanism_family": role.mechanism_family,
                    "allowed_for_candidate_generation": role.allowed_for_candidate_generation,
                    "node_count": len(role.node_ids),
                    "node_ids": json.dumps(list(role.node_ids), sort_keys=True),
                    "construction_phase": "P0_fixed_manifest",
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "role_type", "role_slot", "role_id"]))


def _role_contact_rows(
    weights: dict[tuple[str, str], float],
    roles: tuple[RoleAnnotation, ...],
    module_nodes: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for host in _role_by_type(roles, "host"):
        core = _role_for_slot(roles, role_type="boundary_core", role_slot=host.role_slot)
        host_contact = _contact_weight(weights, module_nodes, host.node_ids)
        core_contact = _contact_weight(weights, module_nodes, _role_nodes(core))
        rows.append(
            {
                "host_slot": host.role_slot,
                "host_contact": host_contact,
                "core_contact": core_contact,
                "core_share": core_contact / host_contact if host_contact else 0.0,
                "core_node_count": len(_role_nodes(core)),
            }
        )
    return rows


def _feature_frame(bundles: list[GraphBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        graph = bundle.canonical_graph
        roles = bundle.roles
        weights = _edge_weight_lookup(graph)
        module_roles = [
            *_role_by_type(roles, "bridge"),
            *_role_by_type(roles, "small_module"),
            *_role_by_type(roles, "middle_module"),
            *_role_by_type(roles, "weak_module"),
        ]
        module_nodes = tuple(sorted({node for role in module_roles for node in role.node_ids}))
        contact_rows = _role_contact_rows(weights, roles, module_nodes)
        contacts = sorted((row["host_contact"] for row in contact_rows), reverse=True)
        top_contact = contacts[0] if contacts else 0.0
        second_contact = contacts[1] if len(contacts) > 1 else 0.0
        total_contact = sum(contacts)
        max_core_share = max((row["core_share"] for row in contact_rows), default=0.0)
        weak_pair_scores = []
        for pair in _role_by_type(roles, "weak_pair"):
            pair_contacts = [
                _contact_weight(weights, pair.node_ids, host.node_ids)
                for host in _role_by_type(roles, "host")
            ]
            pair_total = sum(pair_contacts)
            weak_pair_scores.append(max(pair_contacts) / pair_total if pair_total else 0.0)
        is_control = "control" in bundle.case.mechanism_state
        boundary_core_decoys = _role_by_type(roles, "boundary_core_decoy")
        weak_pair_decoys = _role_by_type(roles, "weak_pair_decoy")
        decoy_roles = [*boundary_core_decoys, *weak_pair_decoys]
        decoy_core_nodes = tuple(sorted({node for role in boundary_core_decoys for node in role.node_ids}))
        decoy_touched_nodes = tuple(sorted({node for role in decoy_roles for node in role.node_ids}))
        decoy_contact_mass = _contact_weight(weights, module_nodes, decoy_core_nodes)
        if not is_control:
            decoy_match_status = "not_applicable_preserved_variant"
        elif decoy_roles:
            decoy_match_status = "explicit_decoy_roles_present"
        else:
            decoy_match_status = "no_explicit_decoy_roles"
        rows.append(
            {
                "variant_id": bundle.case.variant_id,
                "mechanism_family": bundle.case.mechanism_family,
                "mechanism_state": bundle.case.mechanism_state,
                "local_contact_mass": total_contact,
                "cut_gap": top_contact - second_contact,
                "host_dominance": top_contact / total_contact if total_contact else 0.0,
                "boundary_core_concentration": max_core_share,
                "weak_pair_concentration": max(weak_pair_scores, default=0.0),
                "host_contact_profile": json.dumps(_json_safe(contact_rows), sort_keys=True),
                "weak_pair_count": len(_role_by_type(roles, "weak_pair")),
                "candidate_module_node_count": len(module_nodes),
                "decoy_target_rule_family": bundle.case.responsible_rule if is_control and decoy_roles else "",
                "decoy_role_count": len(decoy_roles),
                "decoy_contact_mass": decoy_contact_mass,
                "decoy_touched_node_count": len(decoy_touched_nodes),
                "decoy_match_status": decoy_match_status,
                "control_decoy_required": bool(is_control),
                "control_decoy_available": bool(is_control and decoy_roles),
                "control_decoy_node_count": len(decoy_touched_nodes) if is_control else 0,
                "control_decoy_contact_mass": decoy_contact_mass if is_control else 0.0,
                "control_decoy_waiver": "",
                "construction_phase": "P1_graph_only_diagnostics",
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values("variant_id"))


def _candidate_registry_frame(candidates: list[VariantCandidate], cases: dict[str, VariantGraphCase]) -> pd.DataFrame:
    rows = []
    for candidate in sorted(candidates, key=lambda item: (item.variant_id, item.handle_candidate_id)):
        case = cases[candidate.variant_id]
        rows.append(
            {
                "variant_id": candidate.variant_id,
                "mechanism_family": case.mechanism_family,
                "mechanism_state": case.mechanism_state,
                "handle_candidate_id": candidate.handle_candidate_id,
                "candidate_signature_id": candidate.candidate_signature,
                "candidate_class_signature_id": candidate.candidate_class_signature,
                "handle_type": candidate.handle_type,
                "target_mechanism_class": candidate.target_mechanism_class,
                "target_claim_allowed": candidate.target_claim_allowed,
                "responsible_rule": case.responsible_rule,
                "touched_node_count": len(candidate.touched_nodes),
                "initial_group_count": len(candidate.initial_groups),
                "initial_groups": json.dumps([list(group) for group in candidate.initial_groups], sort_keys=True),
                "initial_role_groups": json.dumps([list(group) for group in candidate.initial_role_groups], sort_keys=True),
                "touched_nodes": json.dumps(list(candidate.touched_nodes), sort_keys=True),
                "evidence_role_ids": json.dumps(list(candidate.evidence_role_ids), sort_keys=True),
                "graph_rule": candidate.graph_rule,
                "graph_evidence": json.dumps(_json_safe(candidate.graph_evidence), sort_keys=True),
                "construction_phase": "P2_blind_candidate_construction",
                "construction_input_policy": "graph_manifest_roles_edges_only",
                "excluded_before_candidate_registry": json.dumps(EXCLUDED_BEFORE_PHASE_LOCK),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows))


def _class_hash(candidates: list[VariantCandidate]) -> str:
    return _sha256_json(sorted(candidate.candidate_class_signature for candidate in candidates))


def _role_invariance_frame(bundles: list[GraphBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        canonical = _generate_candidates(bundle, bundle.roles)
        opaque = _generate_candidates(bundle, bundle.opaque_roles)
        role_permuted = _generate_candidates(bundle, bundle.role_name_permuted_roles)
        canonical_classes = sorted(candidate.candidate_class_signature for candidate in canonical)
        opaque_classes = sorted(candidate.candidate_class_signature for candidate in opaque)
        role_permuted_classes = sorted(candidate.candidate_class_signature for candidate in role_permuted)
        rows.append(
            {
                "variant_id": bundle.case.variant_id,
                "mechanism_family": bundle.case.mechanism_family,
                "canonical_candidate_count": len(canonical),
                "opaque_node_candidate_count": len(opaque),
                "role_name_permuted_candidate_count": len(role_permuted),
                "canonical_class_hash": _class_hash(canonical),
                "opaque_node_class_hash": _class_hash(opaque),
                "role_name_permuted_class_hash": _class_hash(role_permuted),
                "opaque_node_class_match": canonical_classes == opaque_classes,
                "role_name_permutation_match": canonical_classes == role_permuted_classes,
                "raw_node_id_independent": canonical_classes == opaque_classes,
                "candidate_class_match": canonical_classes == opaque_classes == role_permuted_classes,
                "construction_phase": "P3_role_name_invariance",
                "excluded_before_role_invariance": json.dumps(EXCLUDED_BEFORE_PHASE_LOCK),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values("variant_id"))


def _phase_lock_payload(
    *,
    output_dir: Path,
    config: dict[str, Any],
    artifact_names: list[str],
) -> dict[str, Any]:
    artifact_hashes = {
        name: _sha256_file(output_dir / name)
        for name in sorted(artifact_names)
    }
    panel_version = str(config.get("panel_version", "v1"))
    payload = {
        "phase_lock_version": f"tiny_cpm_mechanism_variant_panel_p0_p4_{panel_version}",
        "phase_locked_through": "P4",
        "locked_phases": [
            "P0_fixed_manifest",
            "P1_graph_only_diagnostics",
            "P2_blind_candidate_construction",
            "P3_role_name_invariance",
        ],
        "next_allowed_phase": "P5_baseline_seed_sweep",
        "config": _json_safe(config),
        "artifact_hashes": artifact_hashes,
        "excluded_before_phase_lock": EXCLUDED_BEFORE_PHASE_LOCK,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["phase_lock_hash"] = _sha256_json(
        {
            "phase_lock_version": payload["phase_lock_version"],
            "config": payload["config"],
            "artifact_hashes": artifact_hashes,
            "excluded_before_phase_lock": EXCLUDED_BEFORE_PHASE_LOCK,
        },
        length=None,
    )
    return payload


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.to_numpy().tolist()]
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def _write_report(
    *,
    output_dir: Path,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    registry: pd.DataFrame,
    invariance: pd.DataFrame,
    phase_lock: dict[str, Any],
) -> None:
    family_counts = manifest.groupby("mechanism_family", sort=True)["variant_id"].count().to_dict()
    candidate_counts = registry.groupby("mechanism_family", sort=True)["handle_candidate_id"].count().to_dict()
    control_rows = manifest["mechanism_state"].astype(str).str.contains("control").sum()
    invariance_pass = bool(invariance["candidate_class_match"].all())
    lines = [
        "# Tiny CPM Mechanism Variant Panel P0-P4",
        "",
        "This artifact stops before Leiden seed sweeps or endpoint evaluation.",
        "",
        f"- output_dir: `{_rel(output_dir)}`",
        f"- panel_version: `{phase_lock.get('config', {}).get('panel_version', 'v1')}`",
        f"- variant_count: `{len(manifest)}`",
        f"- control_variant_count: `{int(control_rows)}`",
        f"- mechanism_family_counts: `{json.dumps(family_counts, sort_keys=True)}`",
        f"- blind_candidate_count: `{len(registry)}`",
        f"- blind_candidate_counts: `{json.dumps(candidate_counts, sort_keys=True)}`",
        f"- role_name_invariance_pass: `{invariance_pass}`",
        f"- phase_lock_hash: `{phase_lock['phase_lock_hash']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## P0-P4 Gate Read",
        "",
        "- P0 fixed graph manifest is materialized before execution outcomes.",
        "- P1 mechanism diagnostics use graph, role, and edge evidence only.",
        "- P2 candidate registry is built without frozen endpoints, replay rows, endpoint ranks, method hits, seed outcomes, or restart curves.",
        "- P3 role/name invariance compares canonical node names, opaque node ids, and permuted role labels.",
        "- P4 phase lock hashes all P0-P3 artifacts and runner config.",
        "",
        "## Not Opened",
        "",
        "- baseline reproduction",
        "- endpoint replay stability",
        "- target-scoped method attribution",
        "- restart p75 comparison",
        "- wall/pathway, quality/cost, NanoClustering generality, or algorithm claims",
        "",
        "## Mechanism Feature Snapshot",
        "",
        _markdown_table(
            features[
                [
                    "variant_id",
                    "mechanism_family",
                    "mechanism_state",
                    "local_contact_mass",
                    "cut_gap",
                    "host_dominance",
                    "boundary_core_concentration",
                    "weak_pair_concentration",
                    "decoy_role_count",
                    "decoy_contact_mass",
                    "decoy_match_status",
                ]
            ]
        ),
        "",
    ]
    output_dir.joinpath(REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_p0_p4(
    *,
    output_dir: Path,
    panel_version: str,
    seeds: int,
    n_iterations: int,
    replay_seeds: int,
    baseline_permutations: int,
    budgets: tuple[int, ...],
    recurrent_threshold_share: float,
    recurrent_threshold_min: int,
    strong_baseline_quantile: float,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_lock_path = output_dir / PHASE_LOCK_JSON
    if phase_lock_path.exists() and not force:
        raise FileExistsError(
            f"{_rel(phase_lock_path)} already exists. Use --force only when intentionally regenerating P0-P4."
        )

    config = {
        "output_dir": _rel(output_dir),
        "panel_version": panel_version,
        "seeds": seeds,
        "n_iterations": n_iterations,
        "replay_seeds": replay_seeds,
        "baseline_permutations": baseline_permutations,
        "budgets": list(budgets),
        "recurrent_threshold_share": recurrent_threshold_share,
        "recurrent_threshold_min": recurrent_threshold_min,
        "recurrent_threshold_rule": f"max({recurrent_threshold_min}, ceil(seeds * {recurrent_threshold_share}))",
        "recurrent_threshold_value": max(recurrent_threshold_min, math.ceil(seeds * recurrent_threshold_share)),
        "strong_baseline_quantile": strong_baseline_quantile,
        "phase_scope": "P0-P4 only",
        "excluded_before_phase_lock": EXCLUDED_BEFORE_PHASE_LOCK,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output_dir.joinpath(CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    cases = _variant_cases(seeds, panel_version=panel_version)
    bundles = _materialize_bundles(cases)
    cases_by_id = {case.variant_id: case for case in cases}
    manifest = _manifest_frame(bundles)
    edges = _edges_frame(bundles)
    roles = _roles_frame(bundles)
    features = _feature_frame(bundles)
    candidates = [candidate for bundle in bundles for candidate in _generate_candidates(bundle)]
    registry = _candidate_registry_frame(candidates, cases_by_id)
    invariance = _role_invariance_frame(bundles)

    _write_csv(manifest, output_dir / GRAPH_MANIFEST_CSV)
    _write_csv(edges, output_dir / GRAPH_EDGES_CSV)
    _write_csv(roles, output_dir / GRAPH_ROLES_CSV)
    _write_csv(features, output_dir / MECHANISM_FEATURES_CSV)
    _write_csv(registry, output_dir / BLIND_CANDIDATE_REGISTRY_CSV)
    _write_csv(invariance, output_dir / ROLE_INVARIANCE_CSV)

    artifact_names = [
        CONFIG_JSON,
        GRAPH_MANIFEST_CSV,
        GRAPH_EDGES_CSV,
        GRAPH_ROLES_CSV,
        MECHANISM_FEATURES_CSV,
        BLIND_CANDIDATE_REGISTRY_CSV,
        ROLE_INVARIANCE_CSV,
    ]
    phase_lock = _phase_lock_payload(
        output_dir=output_dir,
        config=config,
        artifact_names=artifact_names,
    )
    phase_lock_path.write_text(
        json.dumps(_json_safe(phase_lock), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        manifest=manifest,
        features=features,
        registry=registry,
        invariance=invariance,
        phase_lock=phase_lock,
    )

    summary = {
        "output_dir": _rel(output_dir),
        "panel_version": panel_version,
        "variant_count": int(len(manifest)),
        "candidate_count": int(len(registry)),
        "role_row_count": int(len(roles)),
        "edge_row_count": int(len(edges)),
        "role_invariance_pass": bool(invariance["candidate_class_match"].all()),
        "phase_lock_hash": phase_lock["phase_lock_hash"],
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            GRAPH_MANIFEST_CSV,
            GRAPH_EDGES_CSV,
            GRAPH_ROLES_CSV,
            MECHANISM_FEATURES_CSV,
            BLIND_CANDIDATE_REGISTRY_CSV,
            ROLE_INVARIANCE_CSV,
            PHASE_LOCK_JSON,
            CONFIG_JSON,
            REPORT_MD,
        ],
    }
    return summary


def _parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not budgets:
        raise argparse.ArgumentTypeError("at least one budget is required")
    if any(budget <= 0 for budget in budgets):
        raise argparse.ArgumentTypeError("budgets must be positive integers")
    return budgets


def _default_output_dir(panel_version: str) -> Path:
    if panel_version == "v1":
        return DEFAULT_OUTPUT_DIR_V1
    if panel_version == "v1_1":
        return DEFAULT_OUTPUT_DIR_V1_1
    if panel_version == "v1_2":
        return DEFAULT_OUTPUT_DIR_V1_2
    raise argparse.ArgumentTypeError(f"unknown panel version: {panel_version}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize Stress 4 tiny CPM mechanism variant panel P0-P4 artifacts."
    )
    parser.add_argument("--panel-version", choices=("v1", "v1_1", "v1_2"), default="v1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--n-iterations", type=int, default=-1)
    parser.add_argument("--replay-seeds", type=int, default=10)
    parser.add_argument("--baseline-permutations", type=int, default=1000)
    parser.add_argument("--budgets", type=_parse_budgets, default=_parse_budgets("1,2,3,5,10,20"))
    parser.add_argument("--recurrent-threshold-share", type=float, default=0.05)
    parser.add_argument("--recurrent-threshold-min", type=int, default=2)
    parser.add_argument("--strong-baseline-quantile", type=float, default=0.75)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir(args.panel_version)

    summary = run_p0_p4(
        output_dir=output_dir,
        panel_version=args.panel_version,
        seeds=args.seeds,
        n_iterations=args.n_iterations,
        replay_seeds=args.replay_seeds,
        baseline_permutations=args.baseline_permutations,
        budgets=args.budgets,
        recurrent_threshold_share=args.recurrent_threshold_share,
        recurrent_threshold_min=args.recurrent_threshold_min,
        strong_baseline_quantile=args.strong_baseline_quantile,
        force=args.force,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
