#!/usr/bin/env python3
"""Run controlled synthetic Leiden+CPM demos for variable-pair mechanisms.

This runner consumes the synthetic-demo design artifact derived from the
NanoClustering variable-pair local ablation. It implements only the fixed
6-family surface:

- stable direct-contact competition;
- partial direct-contact competition;
- coupled negative-direct bridge contact;
- rare start-sensitive direct contact;
- overcompeting bridge-context control;
- nonlocal negative-direct context control.

It uses ordinary Leiden+CPM on small synthetic graphs. It does not run the
NanoClustering graph, execute routes/pathways, promote walls, compare
quality/cost as success claims, or claim a method/algorithm.
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
import numpy as np
import pandas as pd

from sciscape.clustering.runner import LeidenRunner


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_DESIGN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_gamma1e5_20260603"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_variable_pair_synthetic_demo_v1_20260603"
)

DESIGN_FAMILY_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_family_rows.csv"
)
GRAPH_MANIFEST_CSV = "variable_pair_synthetic_demo_graph_manifest.csv"
GRAPH_EDGES_CSV = "variable_pair_synthetic_demo_graph_edges.csv"
SEED_RUNS_CSV = "variable_pair_synthetic_demo_seed_runs.csv"
VARIANT_SUMMARY_CSV = "variable_pair_synthetic_demo_variant_summary.csv"
FAMILY_GATE_ROWS_CSV = "variable_pair_synthetic_demo_family_gate_rows.csv"
SUMMARY_JSON = "variable_pair_synthetic_demo_summary.json"
CONFIG_JSON = "variable_pair_synthetic_demo_config.json"
REPORT_MD = "variable_pair_synthetic_demo_report.md"

GRAPH_VARIANTS = (
    "original",
    "drop_direct_edge",
    "drop_bridge_edges",
    "drop_direct_and_bridge_edges",
)
START_CONDITIONS = (
    "singleton",
    "pair_together",
    "bridges_to_left",
    "bridges_to_right",
    "all_local_together",
)
CLAIM_BOUNDARY = (
    "Variable-pair synthetic CPM demo only; ordinary Leiden+CPM on fixed small "
    "graphs derived from the frozen design surface. No full NanoClustering "
    "replay, no route/pathway execution, no wall promotion, no quality/cost "
    "claim, no method claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_synthetic_demo_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "plain_leiden_cpm_synthetic_diagnostic_not_method"
RUN_STATUS = "executed_variable_pair_synthetic_demo"


@dataclass(frozen=True)
class EdgeSpec:
    left: str
    right: str
    weight: float
    edge_type: str


@dataclass(frozen=True)
class SyntheticCase:
    design_family: str
    synthetic_demo_role: str
    expected_signature: str
    gamma: float
    nodes: tuple[str, ...]
    node_sizes: tuple[int, ...]
    bridge_nodes: tuple[str, ...]
    edges: tuple[EdgeSpec, ...]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["run_status"] = RUN_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _add_edge(
    edges: list[EdgeSpec],
    left: str,
    right: str,
    weight: float,
    edge_type: str,
) -> None:
    if left == right:
        raise ValueError("self edges are not used")
    edges.append(
        EdgeSpec(
            left=min(left, right),
            right=max(left, right),
            weight=float(weight),
            edge_type=str(edge_type),
        )
    )


def _add_clique(edges: list[EdgeSpec], nodes: list[str], weight: float, edge_type: str) -> None:
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            _add_edge(edges, left, right, weight, edge_type)


def _case_common_nodes() -> tuple[list[str], list[str], list[str], list[str]]:
    left_hosts = ["la0", "la1", "la2"]
    right_hosts = ["ra0", "ra1", "ra2"]
    left_bridges = ["lb0", "lb1"]
    right_bridges = ["rb0", "rb1"]
    return left_hosts, right_hosts, left_bridges, right_bridges


def _node_sizes_for(nodes: list[str], *, pair_node_size: int = 1) -> tuple[int, ...]:
    return tuple(int(pair_node_size) if node in {"L", "R"} else 1 for node in nodes)


def _competition_case(
    *,
    design_family: str,
    synthetic_demo_role: str,
    expected_signature: str,
    direct_weight: float,
    pair_bridge_weight: float,
    bridge_host_weight: float,
    host_clique_weight: float,
    cross_context_weight: float = 0.0,
    gamma: float = 1.0,
    pair_node_size: int = 1,
) -> SyntheticCase:
    left_hosts, right_hosts, left_bridges, right_bridges = _case_common_nodes()
    nodes = ["L", "R", *left_hosts, *right_hosts, *left_bridges, *right_bridges]
    edges: list[EdgeSpec] = []
    _add_clique(edges, left_hosts, host_clique_weight, "host_context")
    _add_clique(edges, right_hosts, host_clique_weight, "host_context")
    _add_edge(edges, "L", "R", direct_weight, "direct_pair")
    for bridge in left_bridges:
        _add_edge(edges, "L", bridge, pair_bridge_weight, "pair_bridge")
        for host in left_hosts:
            _add_edge(edges, bridge, host, bridge_host_weight, "bridge_context")
    for bridge in right_bridges:
        _add_edge(edges, "R", bridge, pair_bridge_weight, "pair_bridge")
        for host in right_hosts:
            _add_edge(edges, bridge, host, bridge_host_weight, "bridge_context")
    if cross_context_weight > 0.0:
        for left, right in zip(left_bridges, right_bridges, strict=True):
            _add_edge(edges, left, right, cross_context_weight, "bridge_context")
    return SyntheticCase(
        design_family=design_family,
        synthetic_demo_role=synthetic_demo_role,
        expected_signature=expected_signature,
        gamma=float(gamma),
        nodes=tuple(nodes),
        node_sizes=_node_sizes_for(nodes, pair_node_size=pair_node_size),
        bridge_nodes=tuple([*left_bridges, *right_bridges]),
        edges=tuple(edges),
    )


def _coupled_case() -> SyntheticCase:
    nodes = ["L", "R", "cb0", "cb1"]
    edges: list[EdgeSpec] = []
    _add_edge(edges, "L", "R", 3.40, "direct_pair")
    for bridge in ["cb0", "cb1"]:
        _add_edge(edges, "L", bridge, 2.50, "pair_bridge")
        _add_edge(edges, "R", bridge, 2.50, "pair_bridge")
    _add_edge(edges, "cb0", "cb1", 1.20, "bridge_context")
    return SyntheticCase(
        design_family="coupled_negative_direct_bridge_contact",
        synthetic_demo_role="positive_family_coupled_control",
        expected_signature="original_high_drop_direct_low_drop_bridge_low",
        gamma=1.0,
        nodes=tuple(nodes),
        node_sizes=_node_sizes_for(nodes, pair_node_size=2),
        bridge_nodes=("cb0", "cb1"),
        edges=tuple(edges),
    )


def _overcompeting_case() -> SyntheticCase:
    nodes = ["L", "R", "lb", "rb"]
    edges: list[EdgeSpec] = []
    _add_edge(edges, "L", "R", 4.50, "direct_pair")
    _add_edge(edges, "L", "lb", 3.00, "pair_bridge")
    _add_edge(edges, "R", "rb", 3.00, "pair_bridge")
    return SyntheticCase(
        design_family="overcompeting_bridge_context_control",
        synthetic_demo_role="negative_context_control",
        expected_signature="original_low_drop_direct_low_drop_bridge_high",
        gamma=1.0,
        nodes=tuple(nodes),
        node_sizes=_node_sizes_for(nodes, pair_node_size=2),
        bridge_nodes=("lb", "rb"),
        edges=tuple(edges),
    )


def _nonlocal_control_case() -> SyntheticCase:
    return _competition_case(
        design_family="nonlocal_negative_direct_context_control",
        synthetic_demo_role="negative_locality_control",
        expected_signature="original_low_drop_direct_low_drop_bridge_low",
        direct_weight=0.55,
        pair_bridge_weight=0.55,
        bridge_host_weight=1.55,
        host_clique_weight=1.45,
        gamma=1.0,
    )


def _synthetic_cases(families: pd.DataFrame) -> list[SyntheticCase]:
    wanted = set(families["design_family"].astype(str))
    cases = [
        _competition_case(
            design_family="stable_direct_contact_competition",
            synthetic_demo_role="positive_family_primary",
            expected_signature="original_high_drop_direct_low_drop_bridge_high",
            direct_weight=2.15,
            pair_bridge_weight=0.95,
            bridge_host_weight=1.10,
            host_clique_weight=1.20,
            gamma=1.0,
        ),
        _competition_case(
            design_family="partial_direct_contact_competition",
            synthetic_demo_role="positive_family_boundary",
            expected_signature="original_mixed_drop_direct_low_drop_bridge_high",
            direct_weight=4.25,
            pair_bridge_weight=2.25,
            bridge_host_weight=1.35,
            host_clique_weight=1.25,
            gamma=1.0,
            pair_node_size=2,
        ),
        _coupled_case(),
        _competition_case(
            design_family="rare_start_sensitive_direct_contact",
            synthetic_demo_role="near_boundary_stress_case",
            expected_signature="original_rare_drop_direct_low_drop_bridge_high",
            direct_weight=1.08,
            pair_bridge_weight=1.35,
            bridge_host_weight=1.45,
            host_clique_weight=1.25,
            gamma=1.0,
        ),
        _overcompeting_case(),
        _nonlocal_control_case(),
    ]
    return [case for case in cases if case.design_family in wanted]


def _edges_for_variant(case: SyntheticCase, graph_variant: str) -> tuple[EdgeSpec, ...]:
    output: list[EdgeSpec] = []
    for edge in case.edges:
        if graph_variant in {"drop_direct_edge", "drop_direct_and_bridge_edges"} and edge.edge_type == "direct_pair":
            continue
        if graph_variant in {"drop_bridge_edges", "drop_direct_and_bridge_edges"} and edge.edge_type == "pair_bridge":
            continue
        output.append(edge)
    return tuple(output)


def _build_graph(nodes: tuple[str, ...], edges: tuple[EdgeSpec, ...]) -> ig.Graph:
    index = {name: offset for offset, name in enumerate(nodes)}
    merged: dict[tuple[str, str], float] = defaultdict(float)
    for edge in edges:
        merged[(edge.left, edge.right)] += float(edge.weight)
    edge_names = sorted(merged)
    graph = ig.Graph(
        n=len(nodes),
        edges=[(index[left], index[right]) for left, right in edge_names],
        directed=False,
    )
    graph.vs["name"] = list(nodes)
    graph.es["weight"] = [float(merged[pair]) for pair in edge_names]
    return graph


def _initial_membership(case: SyntheticCase, start_condition: str) -> list[int]:
    index = {node: offset for offset, node in enumerate(case.nodes)}
    membership = list(range(len(case.nodes)))
    if start_condition == "singleton":
        return membership
    if start_condition == "pair_together":
        membership[index["R"]] = membership[index["L"]]
        return _renumber(membership)
    if start_condition == "bridges_to_left":
        label = membership[index["L"]]
        for node in case.bridge_nodes:
            membership[index[node]] = label
        return _renumber(membership)
    if start_condition == "bridges_to_right":
        label = membership[index["R"]]
        for node in case.bridge_nodes:
            membership[index[node]] = label
        return _renumber(membership)
    if start_condition == "all_local_together":
        return [0 for _ in case.nodes]
    raise ValueError(f"unknown start condition: {start_condition}")


def _renumber(membership: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    output: list[int] = []
    next_label = 0
    for label in membership:
        label = int(label)
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        output.append(mapping[label])
    return output


def _canonical_groups(nodes: tuple[str, ...], membership: list[int]) -> list[list[str]]:
    groups: dict[int, list[str]] = {}
    for node, label in zip(nodes, membership, strict=True):
        groups.setdefault(int(label), []).append(str(node))
    return sorted(sorted(group) for group in groups.values())


def _signature_id(groups: list[list[str]]) -> str:
    payload = json.dumps(groups, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _mechanism_read(case: SyntheticCase, membership: list[int]) -> dict[str, Any]:
    labels = {node: int(label) for node, label in zip(case.nodes, membership, strict=True)}
    left_label = labels["L"]
    right_label = labels["R"]
    pair_coassigned = left_label == right_label
    left_bridge_count = sum(1 for node in case.bridge_nodes if labels[node] == left_label)
    right_bridge_count = sum(1 for node in case.bridge_nodes if labels[node] == right_label)
    if pair_coassigned and left_bridge_count > 0:
        read = "pair_coassigned_with_bridge_context"
    elif pair_coassigned:
        read = "pair_coassigned_without_bridge_context"
    elif left_bridge_count > 0 and right_bridge_count > 0:
        read = "pair_separated_bridge_split"
    elif left_bridge_count > 0 or right_bridge_count > 0:
        read = "pair_separated_single_side_bridge"
    else:
        read = "pair_separated_no_bridge_context"
    return {
        "pair_coassigned": bool(pair_coassigned),
        "left_bridge_same_cluster_count": int(left_bridge_count),
        "right_bridge_same_cluster_count": int(right_bridge_count),
        "mechanism_read": read,
    }


def _graph_manifest_and_edges(cases: list[SyntheticCase]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for case in cases:
        for graph_variant in GRAPH_VARIANTS:
            edges = _edges_for_variant(case, graph_variant)
            graph = _build_graph(case.nodes, edges)
            manifest_rows.append(
                {
                    "design_family": case.design_family,
                    "synthetic_demo_role": case.synthetic_demo_role,
                    "expected_signature": case.expected_signature,
                    "graph_variant": graph_variant,
                    "gamma": float(case.gamma),
                    "node_count": int(graph.vcount()),
                    "node_size_sum": int(sum(case.node_sizes)),
                    "node_sizes": ";".join(
                        f"{node}:{size}" for node, size in zip(case.nodes, case.node_sizes, strict=True)
                    ),
                    "edge_count": int(graph.ecount()),
                    "edge_weight_sum": float(sum(graph.es["weight"])) if graph.ecount() else 0.0,
                    "bridge_nodes": ";".join(case.bridge_nodes),
                }
            )
            for edge in edges:
                edge_rows.append(
                    {
                        "design_family": case.design_family,
                        "graph_variant": graph_variant,
                        "source": edge.left,
                        "target": edge.right,
                        "weight": float(edge.weight),
                        "edge_type": edge.edge_type,
                    }
                )
    return _claim_columns(pd.DataFrame(manifest_rows)), _claim_columns(pd.DataFrame(edge_rows))


def _run_seed_sweeps(
    *,
    cases: list[SyntheticCase],
    seeds: int,
    n_iterations: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for graph_variant in GRAPH_VARIANTS:
            graph = _build_graph(case.nodes, _edges_for_variant(case, graph_variant))
            runner = LeidenRunner(graph, objective="cpm", default_iterations=int(n_iterations))
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
                    rows.append(
                        {
                            "design_family": case.design_family,
                            "synthetic_demo_role": case.synthetic_demo_role,
                            "expected_signature": case.expected_signature,
                            "graph_variant": graph_variant,
                            "start_condition": start_condition,
                            "seed": int(seed),
                            "gamma": float(case.gamma),
                            "n_iterations": int(n_iterations),
                            "node_count": int(graph.vcount()),
                            "edge_count": int(graph.ecount()),
                            "cluster_count": int(result.cluster_count),
                            "quality": float(result.quality),
                            "endpoint_signature_id": _signature_id(groups),
                            "endpoint_signature": json.dumps(groups, sort_keys=True),
                            **_mechanism_read(case, membership),
                        }
                    )
    return _claim_columns(pd.DataFrame(rows))


def _variant_summary(seed_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "design_family",
        "synthetic_demo_role",
        "expected_signature",
        "graph_variant",
    ]
    for keys, group in seed_runs.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        endpoint_counts = group["endpoint_signature_id"].value_counts()
        pair_coassigned = group["pair_coassigned"].astype(bool)
        rows.append(
            {
                **key_data,
                "run_count": int(len(group)),
                "start_condition_count": int(group["start_condition"].nunique()),
                "seed_count": int(group["seed"].nunique()),
                "distinct_endpoint_count": int(endpoint_counts.shape[0]),
                "top_endpoint_share": float(endpoint_counts.max() / len(group)),
                "pair_coassigned_run_count": int(pair_coassigned.sum()),
                "pair_coassigned_share": float(pair_coassigned.mean()),
                "quality_min": float(group["quality"].min()),
                "quality_median": float(group["quality"].median()),
                "quality_max": float(group["quality"].max()),
                "cluster_count_min": int(group["cluster_count"].min()),
                "cluster_count_median": float(group["cluster_count"].median()),
                "cluster_count_max": int(group["cluster_count"].max()),
                "mechanism_read_counts": json.dumps(
                    group["mechanism_read"].value_counts().to_dict(),
                    sort_keys=True,
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _family_gate_rows(variant_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in variant_summary.groupby("design_family", sort=False):
        by_variant = group.set_index("graph_variant")

        def share(variant: str) -> float:
            if variant not in by_variant.index:
                return 0.0
            return float(by_variant.loc[variant, "pair_coassigned_share"])

        original = share("original")
        drop_direct = share("drop_direct_edge")
        drop_bridge = share("drop_bridge_edges")
        drop_both = share("drop_direct_and_bridge_edges")
        expected = str(group["expected_signature"].iloc[0])
        if expected == "original_high_drop_direct_low_drop_bridge_high":
            passed = original >= 0.80 and drop_direct <= 0.10 and drop_bridge >= 0.80 and drop_both <= 0.10
        elif expected == "original_mixed_drop_direct_low_drop_bridge_high":
            passed = 0.15 <= original <= 0.75 and drop_direct <= 0.10 and drop_bridge >= 0.80 and drop_both <= 0.10
        elif expected == "original_high_drop_direct_low_drop_bridge_low":
            passed = original >= 0.80 and drop_direct <= 0.10 and drop_bridge <= 0.10 and drop_both <= 0.10
        elif expected == "original_rare_drop_direct_low_drop_bridge_high":
            passed = 0.0 < original <= 0.25 and drop_direct <= 0.10 and drop_bridge >= 0.80 and drop_both <= 0.10
        elif expected == "original_low_drop_direct_low_drop_bridge_high":
            passed = original <= 0.10 and drop_direct <= 0.10 and drop_bridge >= 0.80 and drop_both <= 0.10
        elif expected == "original_low_drop_direct_low_drop_bridge_low":
            passed = original <= 0.10 and drop_direct <= 0.10 and drop_bridge <= 0.10 and drop_both <= 0.10
        else:
            passed = False
        rows.append(
            {
                "design_family": str(family),
                "synthetic_demo_role": str(group["synthetic_demo_role"].iloc[0]),
                "expected_signature": expected,
                "original_pair_coassigned_share": original,
                "drop_direct_pair_coassigned_share": drop_direct,
                "drop_bridge_pair_coassigned_share": drop_bridge,
                "drop_direct_and_bridge_pair_coassigned_share": drop_both,
                "signature_reproduced": bool(passed),
                "gate_status": (
                    "synthetic_signature_reproduced"
                    if passed
                    else "synthetic_signature_not_reproduced"
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _build_summary(
    *,
    design_dir: Path,
    output_dir: Path,
    seed_runs: pd.DataFrame,
    family_gate_rows: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "variable_pair_synthetic_demo_summary.v1",
        "status": RUN_STATUS,
        "design_dir": str(design_dir),
        "output_dir": str(output_dir),
        "family_count": int(family_gate_rows["design_family"].nunique()),
        "seed_run_count": int(len(seed_runs)),
        "signature_reproduced_count": int(family_gate_rows["signature_reproduced"].astype(bool).sum()),
        "signature_not_reproduced_count": int((~family_gate_rows["signature_reproduced"].astype(bool)).sum()),
        "gate_status_counts": family_gate_rows["gate_status"].value_counts().to_dict(),
        "family_gate_rows": family_gate_rows.to_dict("records"),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    family_gate_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic CPM Demo",
        "",
        f"- status: `{summary['status']}`",
        f"- family_count: {summary['family_count']}",
        f"- seed_run_count: {summary['seed_run_count']}",
        f"- signature_reproduced_count: {summary['signature_reproduced_count']}",
        f"- signature_not_reproduced_count: {summary['signature_not_reproduced_count']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Family Gates",
    ]
    for row in family_gate_rows.itertuples(index=False):
        lines.append(
            "- "
            f"{row.design_family}: status={row.gate_status}, "
            f"orig={row.original_pair_coassigned_share:.3f}, "
            f"drop_direct={row.drop_direct_pair_coassigned_share:.3f}, "
            f"drop_bridge={row.drop_bridge_pair_coassigned_share:.3f}, "
            f"drop_both={row.drop_direct_and_bridge_pair_coassigned_share:.3f}, "
            f"expected={row.expected_signature}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This runner tests whether the fixed 6-family synthetic graph "
                "surface reproduces the local ablation signatures under ordinary "
                "Leiden+CPM. It is not a full-graph replay, route/pathway, wall, "
                "quality/cost, method, or algorithm claim."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    design_dir = Path(args.design_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    families = pd.read_csv(design_dir / DESIGN_FAMILY_ROWS_CSV)
    cases = _synthetic_cases(families)
    graph_manifest, graph_edges = _graph_manifest_and_edges(cases)
    seed_runs = _run_seed_sweeps(
        cases=cases,
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
    )
    variant_summary = _variant_summary(seed_runs)
    family_gate_rows = _family_gate_rows(variant_summary)
    _write_csv(graph_manifest, output_dir / GRAPH_MANIFEST_CSV)
    _write_csv(graph_edges, output_dir / GRAPH_EDGES_CSV)
    _write_csv(seed_runs, output_dir / SEED_RUNS_CSV)
    _write_csv(variant_summary, output_dir / VARIANT_SUMMARY_CSV)
    _write_csv(family_gate_rows, output_dir / FAMILY_GATE_ROWS_CSV)
    summary = _build_summary(
        design_dir=design_dir,
        output_dir=output_dir,
        seed_runs=seed_runs,
        family_gate_rows=family_gate_rows,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_demo.v1",
        "design_dir": str(design_dir),
        "output_dir": str(output_dir),
        "seeds": int(args.seeds),
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
        family_gate_rows=family_gate_rows,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--n-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
