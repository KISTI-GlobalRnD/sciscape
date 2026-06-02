#!/usr/bin/env python3
"""Run tiny Leiden + CPM seed sweeps for basin-like endpoint alternatives.

This is a controlled demo scaffold for Track C. It uses ordinary
igraph/leidenalg CPM only. It does not use a custom transition operator, route
execution, wall promotion, quality/cost selection, or directed search.
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
from typing import Any, Callable

import igraph as ig
import pandas as pd

from sciscape.clustering.runner import LeidenRunner


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_demo_seed_sweep_20260531"

SEED_RUNS_CSV = "leiden_cpm_tiny_demo_seed_runs.csv"
ENDPOINT_SUMMARY_CSV = "leiden_cpm_tiny_demo_endpoint_summary.csv"
FROZEN_ENDPOINT_MANIFEST_CSV = "leiden_cpm_tiny_demo_frozen_endpoint_manifest.csv"
DISCOVERY_CURVE_CSV = "leiden_cpm_tiny_demo_discovery_curve.csv"
FAMILY_SUMMARY_CSV = "leiden_cpm_tiny_demo_family_summary.csv"
GATE_MATRIX_CSV = "leiden_cpm_tiny_demo_gate_matrix.csv"
SUMMARY_JSON = "leiden_cpm_tiny_demo_summary.json"
CONFIG_JSON = "leiden_cpm_tiny_demo_config.json"
REPORT_MD = "leiden_cpm_tiny_demo_report.md"

CLAIM_BOUNDARY = (
    "Tiny CPM demo only; ordinary Leiden + CPM seed sweep, no custom method, "
    "no route execution, no wall/pathway promotion, no basin-quality claim, "
    "no cost claim, and no NanoClustering generality claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_seed_sweep_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
METHOD_STATUS = "baseline_only_no_custom_method"
DISCOVERY_BUDGETS = (1, 2, 3, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class GraphCase:
    family: str
    mechanism: str
    gamma: float
    expected_behavior: str
    builder: Callable[[], ig.Graph]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _add_edge(edges: dict[tuple[str, str], float], left: str, right: str, weight: float = 1.0) -> None:
    if left == right:
        raise ValueError("self edges are not used in tiny demo graphs")
    key = tuple(sorted((left, right)))
    edges[key] += float(weight)


def _add_clique(edges: dict[tuple[str, str], float], nodes: list[str], weight: float = 1.0) -> None:
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            _add_edge(edges, left, right, weight)


def _build_graph(names: list[str], edges: dict[tuple[str, str], float]) -> ig.Graph:
    index = {name: offset for offset, name in enumerate(names)}
    edge_names = sorted(edges)
    graph = ig.Graph(
        n=len(names),
        edges=[(index[left], index[right]) for left, right in edge_names],
        directed=False,
    )
    graph.vs["name"] = names
    graph.es["weight"] = [float(edges[pair]) for pair in edge_names]
    return graph


def _near_tie_bridge_cliques() -> ig.Graph:
    names = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)] + ["x"]
    edges: dict[tuple[str, str], float] = defaultdict(float)
    _add_clique(edges, [f"a{i}" for i in range(5)])
    _add_clique(edges, [f"b{i}" for i in range(5)])
    for offset in range(4):
        _add_edge(edges, "x", f"a{offset}")
        _add_edge(edges, "x", f"b{offset}")
    return _build_graph(names, edges)


def _absorption_triad() -> ig.Graph:
    names = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(6)] + [f"s{i}" for i in range(3)]
    edges: dict[tuple[str, str], float] = defaultdict(float)
    _add_clique(edges, [f"a{i}" for i in range(6)])
    _add_clique(edges, [f"b{i}" for i in range(6)])
    _add_clique(edges, [f"s{i}" for i in range(3)])
    for source in range(3):
        for offset in range(4):
            _add_edge(edges, f"s{source}", f"a{offset}")
        for offset in range(2):
            _add_edge(edges, f"s{source}", f"b{offset}")
    for offset in range(2):
        _add_edge(edges, f"a{offset}", f"b{offset}")
    return _build_graph(names, edges)


def _balanced_split_module() -> ig.Graph:
    names = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)] + [f"m{i}" for i in range(4)]
    edges: dict[tuple[str, str], float] = defaultdict(float)
    _add_clique(edges, [f"a{i}" for i in range(5)])
    _add_clique(edges, [f"b{i}" for i in range(5)])
    _add_clique(edges, [f"m{i}" for i in range(4)])
    for middle in [0, 1]:
        for offset in range(4):
            _add_edge(edges, f"m{middle}", f"a{offset}")
        for offset in range(2):
            _add_edge(edges, f"m{middle}", f"b{offset}")
    for middle in [2, 3]:
        for offset in range(2):
            _add_edge(edges, f"m{middle}", f"a{offset}")
        for offset in range(4):
            _add_edge(edges, f"m{middle}", f"b{offset}")
    return _build_graph(names, edges)


def _diffuse_fragment_star() -> ig.Graph:
    names: list[str] = []
    for host in range(4):
        names.extend(f"h{host}_{offset}" for offset in range(4))
    names.extend(f"x{offset}" for offset in range(4))
    edges: dict[tuple[str, str], float] = defaultdict(float)
    for host in range(4):
        _add_clique(edges, [f"h{host}_{offset}" for offset in range(4)])
    _add_clique(edges, [f"x{offset}" for offset in range(4)])
    for source in range(4):
        for offset in range(3):
            _add_edge(edges, f"x{source}", f"h{source}_{offset}")
        for offset in range(2):
            _add_edge(edges, f"x{source}", f"h{(source + 1) % 4}_{offset}")
    return _build_graph(names, edges)


def _graph_cases() -> list[GraphCase]:
    return [
        GraphCase(
            family="near_tie_bridge_cliques",
            mechanism="near_tie_cpm_cut_bridge_ambiguity",
            gamma=0.50,
            expected_behavior="bridge node x recurrently joins either clique with equal CPM quality",
            builder=_near_tie_bridge_cliques,
        ),
        GraphCase(
            family="absorption_triad",
            mechanism="external_host_absorption",
            gamma=0.75,
            expected_behavior="small module can remain separate or be absorbed by a host-side boundary",
            builder=_absorption_triad,
        ),
        GraphCase(
            family="balanced_split_module",
            mechanism="source_host_balanced_split",
            gamma=0.65,
            expected_behavior="middle module can split across both hosts or collapse into one side",
            builder=_balanced_split_module,
        ),
        GraphCase(
            family="diffuse_fragment_star",
            mechanism="diffuse_multiway_fragmentation",
            gamma=0.65,
            expected_behavior="weak module nodes fragment across several host communities",
            builder=_diffuse_fragment_star,
        ),
    ]


def _canonical_groups(graph: ig.Graph, membership: list[int]) -> list[list[str]]:
    groups: dict[int, list[str]] = {}
    for node_name, label in zip(graph.vs["name"], membership):
        groups.setdefault(int(label), []).append(str(node_name))
    return sorted(sorted(nodes) for nodes in groups.values())


def _signature_id(groups: list[list[str]]) -> str:
    payload = json.dumps(groups, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _membership_by_node(graph: ig.Graph, membership: list[int]) -> dict[str, int]:
    return {str(node_name): int(label) for node_name, label in zip(graph.vs["name"], membership)}


def _nodes_in_same_cluster(
    graph: ig.Graph,
    membership: list[int],
    node_name: str,
) -> list[str]:
    by_node = _membership_by_node(graph, membership)
    label = by_node[node_name]
    return sorted(name for name, cluster in by_node.items() if cluster == label)


def _has_prefix(nodes: list[str], prefix: str) -> bool:
    return any(node.startswith(prefix) for node in nodes)


def _classify_mechanism(case: str, graph: ig.Graph, membership: list[int]) -> str:
    if case == "near_tie_bridge_cliques":
        peers = _nodes_in_same_cluster(graph, membership, "x")
        has_a = _has_prefix(peers, "a")
        has_b = _has_prefix(peers, "b")
        if has_a and not has_b:
            return "bridge_to_a"
        if has_b and not has_a:
            return "bridge_to_b"
        if not has_a and not has_b:
            return "bridge_separate"
        return "bridge_between_both_hosts"

    if case == "absorption_triad":
        s_clusters = {_membership_by_node(graph, membership)[f"s{i}"] for i in range(3)}
        if len(s_clusters) > 1:
            return "small_module_split"
        peers = _nodes_in_same_cluster(graph, membership, "s0")
        has_a = _has_prefix(peers, "a")
        has_b = _has_prefix(peers, "b")
        if has_a and not has_b:
            return "small_module_absorbed_by_a"
        if has_b and not has_a:
            return "small_module_absorbed_by_b"
        if not has_a and not has_b:
            return "small_module_separate"
        return "small_module_between_both_hosts"

    if case == "balanced_split_module":
        by_node = _membership_by_node(graph, membership)
        m_clusters = {f"m{i}": by_node[f"m{i}"] for i in range(4)}
        contexts = {
            node: _nodes_in_same_cluster(graph, membership, node)
            for node in m_clusters
        }
        a_side = all(_has_prefix(contexts[f"m{i}"], "a") for i in [0, 1])
        b_side = all(_has_prefix(contexts[f"m{i}"], "b") for i in [2, 3])
        if a_side and b_side and m_clusters["m0"] == m_clusters["m1"] and m_clusters["m2"] == m_clusters["m3"]:
            if m_clusters["m0"] != m_clusters["m2"]:
                return "balanced_middle_split"
        if len(set(m_clusters.values())) == 1:
            peers = contexts["m0"]
            if not _has_prefix(peers, "a") and not _has_prefix(peers, "b"):
                return "middle_module_separate"
            return "middle_module_absorbed_or_merged"
        return "mixed_middle_fragmentation"

    if case == "diffuse_fragment_star":
        by_node = _membership_by_node(graph, membership)
        host_targets: set[str] = set()
        x_clusters = {f"x{i}": by_node[f"x{i}"] for i in range(4)}
        for node in x_clusters:
            peers = _nodes_in_same_cluster(graph, membership, node)
            for host in range(4):
                if _has_prefix(peers, f"h{host}_"):
                    host_targets.add(f"h{host}")
        if len(set(x_clusters.values())) == 1:
            if host_targets:
                return "weak_module_absorbed_or_merged"
            return "weak_module_separate"
        if len(host_targets) >= 2:
            return "diffuse_host_fragmentation"
        return "mixed_weak_module_fragmentation"

    return "unclassified"


def _run_case(case: GraphCase, *, seeds: int, n_iterations: int) -> tuple[pd.DataFrame, ig.Graph]:
    graph = case.builder()
    runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
    rows: list[dict[str, Any]] = []
    for seed in range(seeds):
        result = runner.run(case.gamma, seed=seed)
        membership = list(map(int, result.membership))
        groups = _canonical_groups(graph, membership)
        rows.append(
            {
                "family": case.family,
                "mechanism": case.mechanism,
                "expected_behavior": case.expected_behavior,
                "gamma": case.gamma,
                "seed": seed,
                "node_count": graph.vcount(),
                "edge_count": graph.ecount(),
                "cluster_count": result.cluster_count,
                "quality": float(result.quality),
                "endpoint_signature_id": _signature_id(groups),
                "endpoint_signature": json.dumps(groups, sort_keys=True),
                "mechanism_read": _classify_mechanism(case.family, graph, membership),
            }
        )
    return pd.DataFrame(rows), graph


def _endpoint_summary(seed_runs: pd.DataFrame, *, seeds: int) -> pd.DataFrame:
    rows = (
        seed_runs.groupby(
            [
                "family",
                "mechanism",
                "expected_behavior",
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
    rows["is_recurrent_endpoint"] = rows["seed_count"].ge(max(2, math.ceil(seeds * 0.05)))
    rows = rows.sort_values(
        ["family", "seed_count", "quality_median", "endpoint_signature_id"],
        ascending=[True, False, False, True],
    )
    rows["endpoint_rank_in_family"] = rows.groupby("family").cumcount() + 1
    return _claim_columns(rows)


def _family_summary(endpoint_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in endpoint_summary.groupby("family", sort=True):
        recurrent = group[group["is_recurrent_endpoint"].astype(bool)]
        quality_max = float(group["quality_max"].max())
        top_quality = group[group["quality_max"].ge(quality_max - 1e-9)]
        rows.append(
            {
                "family": str(family),
                "mechanism": str(group["mechanism"].iloc[0]),
                "gamma": float(group["gamma"].iloc[0]),
                "node_count": int(group["node_count"].iloc[0]),
                "edge_count": int(group["edge_count"].iloc[0]),
                "distinct_endpoint_count": int(group["endpoint_signature_id"].nunique()),
                "recurrent_endpoint_count": int(recurrent["endpoint_signature_id"].nunique()),
                "top_quality_endpoint_count": int(top_quality["endpoint_signature_id"].nunique()),
                "max_endpoint_seed_share": float(group["seed_share"].max()),
                "quality_min": float(group["quality_min"].min()),
                "quality_max": quality_max,
                "quality_range": float(quality_max - group["quality_min"].min()),
                "dominant_mechanism_reads": ";".join(
                    group.sort_values("seed_count", ascending=False)["mechanism_read"]
                    .head(4)
                    .astype(str)
                    .tolist()
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows).sort_values("family"))


def _frozen_endpoint_manifest(endpoint_summary: pd.DataFrame) -> pd.DataFrame:
    rows = endpoint_summary.copy()
    rows["frozen_endpoint_id"] = [
        f"{family}__endpoint{int(rank):02d}"
        for family, rank in zip(rows["family"], rows["endpoint_rank_in_family"])
    ]
    rows["freeze_status"] = "frozen_baseline_endpoint_universe_v1"
    rows["freeze_scope"] = (
        "plain_leiden_cpm_tiny_demo;graph_family_gamma_seed_protocol_fixed"
    )
    rows["baseline_role"] = rows["is_recurrent_endpoint"].map(
        {True: "recurrent_baseline_endpoint", False: "rare_baseline_endpoint"}
    )
    preferred = [
        "frozen_endpoint_id",
        "freeze_status",
        "freeze_scope",
        "baseline_role",
        "family",
        "mechanism",
        "gamma",
        "endpoint_rank_in_family",
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
    return _claim_columns(rows[preferred])


def _discovery_curve(
    seed_runs: pd.DataFrame,
    endpoint_manifest: pd.DataFrame,
    *,
    seeds: int,
    permutations: int,
    rng_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    budgets = [budget for budget in DISCOVERY_BUDGETS if budget <= seeds]
    for family, group in seed_runs.groupby("family", sort=True):
        family = str(family)
        seed_to_endpoint = {
            int(row.seed): str(row.endpoint_signature_id)
            for row in group[["seed", "endpoint_signature_id"]].itertuples(index=False)
        }
        seed_values = sorted(seed_to_endpoint)
        manifest = endpoint_manifest[endpoint_manifest["family"].eq(family)]
        recurrent_ids = set(
            manifest.loc[manifest["is_recurrent_endpoint"].astype(bool), "endpoint_signature_id"]
            .astype(str)
            .tolist()
        )
        top_quality = float(manifest["quality_max"].max())
        top_quality_ids = set(
            manifest.loc[
                manifest["quality_max"].ge(top_quality - 1e-9), "endpoint_signature_id"
            ]
            .astype(str)
            .tolist()
        )
        total_ids = set(manifest["endpoint_signature_id"].astype(str).tolist())
        for budget in budgets:
            endpoint_counts: list[int] = []
            recurrent_recalls: list[float] = []
            all_recurrent_hits: list[bool] = []
            top_quality_hits: list[bool] = []
            all_endpoint_hits: list[bool] = []
            for permutation in range(permutations):
                rng = random.Random(
                    f"{rng_seed}:{family}:{budget}:{permutation}:{len(seed_values)}"
                )
                order = seed_values[:]
                rng.shuffle(order)
                found = {seed_to_endpoint[seed] for seed in order[:budget]}
                endpoint_counts.append(len(found))
                if recurrent_ids:
                    recurrent_recalls.append(len(found & recurrent_ids) / len(recurrent_ids))
                    all_recurrent_hits.append(recurrent_ids.issubset(found))
                else:
                    recurrent_recalls.append(1.0)
                    all_recurrent_hits.append(True)
                top_quality_hits.append(bool(found & top_quality_ids))
                all_endpoint_hits.append(total_ids.issubset(found))
            rows.append(
                {
                    "family": family,
                    "budget": int(budget),
                    "permutations": int(permutations),
                    "distinct_endpoint_count_mean": float(
                        sum(endpoint_counts) / len(endpoint_counts)
                    ),
                    "distinct_endpoint_count_min": int(min(endpoint_counts)),
                    "distinct_endpoint_count_max": int(max(endpoint_counts)),
                    "recurrent_endpoint_recall_mean": float(
                        sum(recurrent_recalls) / len(recurrent_recalls)
                    ),
                    "recurrent_endpoint_recall_min": float(min(recurrent_recalls)),
                    "recurrent_endpoint_recall_max": float(max(recurrent_recalls)),
                    "all_recurrent_endpoint_hit_rate": float(
                        sum(all_recurrent_hits) / len(all_recurrent_hits)
                    ),
                    "top_quality_endpoint_hit_rate": float(
                        sum(top_quality_hits) / len(top_quality_hits)
                    ),
                    "all_endpoint_hit_rate": float(
                        sum(all_endpoint_hits) / len(all_endpoint_hits)
                    ),
                    "recurrent_endpoint_count": int(len(recurrent_ids)),
                    "total_endpoint_count": int(len(total_ids)),
                    "rng_seed": int(rng_seed),
                }
            )
    return _claim_columns(pd.DataFrame(rows).sort_values(["family", "budget"]))


def _gate_matrix(family_summary: pd.DataFrame, discovery_curve: pd.DataFrame) -> pd.DataFrame:
    family_count = int(family_summary["family"].nunique())
    recurrent_multi = family_summary[
        family_summary["recurrent_endpoint_count"].ge(2)
    ]
    top_quality_multi = family_summary[
        family_summary["top_quality_endpoint_count"].ge(2)
    ]
    budget20 = discovery_curve[discovery_curve["budget"].eq(20)]
    min_budget20_recurrent_recall = (
        float(budget20["recurrent_endpoint_recall_mean"].min())
        if not budget20.empty
        else 0.0
    )
    rows = [
        {
            "gate_id": "D1_tiny_demo_executed",
            "gate_question": "Were the four predeclared tiny CPM graph families executed?",
            "evidence": f"executed_families={family_count}",
            "status": "pass" if family_count == 4 else "blocked_incomplete_demo_grid",
            "decision": "use_as_controlled_demo_surface",
            "next_action": "inspect endpoint signatures and mechanism reads",
        },
        {
            "gate_id": "D2_plain_leiden_cpm_multi_endpoint",
            "gate_question": "Does ordinary Leiden + CPM produce recurring alternative endpoints?",
            "evidence": f"families_with_recurrent_multi_endpoint={len(recurrent_multi)}",
            "status": "pass" if len(recurrent_multi) >= 1 else "blocked_no_recurring_alternatives",
            "decision": "baseline_can_reproduce_basin_like_endpoint_alternatives",
            "next_action": "use these families for baseline-vs-method comparison design",
        },
        {
            "gate_id": "D3_mechanism_diversity",
            "gate_question": "Do recurring alternatives appear across multiple mechanism families?",
            "evidence": f"families_with_recurrent_multi_endpoint={len(recurrent_multi)}",
            "status": "pass" if len(recurrent_multi) >= 3 else "caveat_required",
            "decision": "controlled_demo_is_not_single_case_only",
            "next_action": "select the smallest clean family for visual/manual explanation",
        },
        {
            "gate_id": "D4_equal_quality_tie_surface",
            "gate_question": "Is there at least one equal-quality multi-endpoint CPM surface?",
            "evidence": f"families_with_multiple_top_quality_endpoints={len(top_quality_multi)}",
            "status": "pass" if len(top_quality_multi) >= 1 else "caveat_required",
            "decision": "near_tie_endpoint_surface_available_for_wall_diagnostics",
            "next_action": "turn endpoint alternatives into route/pathway probes only after schema",
        },
        {
            "gate_id": "D5_method_claim_gate",
            "gate_question": "Can this demo claim our method improves over Leiden + CPM?",
            "evidence": "baseline seed sweep only",
            "status": "closed_excluded_by_design",
            "decision": "keep_method_claims_closed",
            "next_action": "add custom method only after baseline endpoint universe is frozen",
        },
        {
            "gate_id": "D6_baseline_discovery_curve_frozen",
            "gate_question": "Is there a frozen random-restart discovery baseline?",
            "evidence": (
                f"budget20_min_mean_recurrent_recall="
                f"{min_budget20_recurrent_recall:.6f}"
            ),
            "status": "pass" if min_budget20_recurrent_recall > 0 else "blocked_no_curve",
            "decision": "use_curve_as_baseline_for_future_method_comparison",
            "next_action": "compare candidate methods against fixed budget curves",
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix["claim_boundary"] = CLAIM_BOUNDARY
    return matrix


def _write_graph_sidecars(graphs: dict[str, ig.Graph], output_dir: Path) -> None:
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    for family, graph in graphs.items():
        nodes = pd.DataFrame(
            {
                "node_id": list(range(graph.vcount())),
                "node_name": graph.vs["name"],
            }
        )
        edges = pd.DataFrame(
            {
                "source": [graph.vs[edge.source]["name"] for edge in graph.es],
                "target": [graph.vs[edge.target]["name"] for edge in graph.es],
                "weight": graph.es["weight"],
            }
        )
        _write_csv(nodes, graph_dir / f"{family}_nodes.csv")
        _write_csv(edges, graph_dir / f"{family}_edges.csv")


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
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
    family_summary: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
    endpoint_manifest: pd.DataFrame,
    discovery_curve: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    text = [
        "# Tiny Leiden + CPM Demo Seed Sweep",
        "",
        f"- seed_count_per_family: `{summary['seed_count_per_family']}`",
        f"- family_count: `{summary['family_count']}`",
        f"- families_with_recurrent_multi_endpoint: `{summary['families_with_recurrent_multi_endpoint']}`",
        f"- families_with_multiple_top_quality_endpoints: `{summary['families_with_multiple_top_quality_endpoints']}`",
        f"- total_distinct_endpoints: `{summary['total_distinct_endpoints']}`",
        f"- discovery_curve_rows: `{summary['discovery_curve_rows']}`",
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
        "## Family Summary",
        "",
        _markdown_table(
            family_summary,
            [
                "family",
                "mechanism",
                "gamma",
                "distinct_endpoint_count",
                "recurrent_endpoint_count",
                "top_quality_endpoint_count",
                "max_endpoint_seed_share",
                "quality_range",
                "dominant_mechanism_reads",
            ],
            max_rows=10,
        ),
        "",
        "## Endpoint Summary",
        "",
        _markdown_table(
            endpoint_summary.sort_values(
                ["family", "endpoint_rank_in_family"],
                ascending=[True, True],
            ),
            [
                "family",
                "endpoint_rank_in_family",
                "mechanism_read",
                "seed_count",
                "seed_share",
                "quality_median",
                "cluster_count_min",
                "cluster_count_max",
                "endpoint_signature_id",
            ],
            max_rows=32,
        ),
        "",
        "## Frozen Endpoint Manifest",
        "",
        _markdown_table(
            endpoint_manifest.sort_values(["family", "endpoint_rank_in_family"]),
            [
                "frozen_endpoint_id",
                "baseline_role",
                "family",
                "mechanism_read",
                "seed_count",
                "seed_share",
                "quality_median",
            ],
            max_rows=32,
        ),
        "",
        "## Discovery Curve",
        "",
        _markdown_table(
            discovery_curve[discovery_curve["budget"].isin([5, 10, 20, 50, 100])],
            [
                "family",
                "budget",
                "distinct_endpoint_count_mean",
                "recurrent_endpoint_recall_mean",
                "all_recurrent_endpoint_hit_rate",
                "top_quality_endpoint_hit_rate",
            ],
            max_rows=32,
        ),
        "",
        "## Read",
        "",
        "- This is the first controlled baseline surface: no NanoClustering data and no custom method.",
        "- Passing D2/D3 means the guiding premise can now be demonstrated in tiny CPM graphs before method claims.",
        "- D4 identifies near-tie/equal-quality endpoint surfaces suitable for later route/pathway schema work.",
        "- D6 freezes the random-restart discovery baseline that future candidate methods must beat.",
        "- D5 remains closed: method improvement needs a separate baseline-vs-method runner.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def run_demo(
    *,
    output_dir: Path,
    seeds: int,
    n_iterations: int,
    discovery_permutations: int,
    discovery_rng_seed: int,
) -> dict[str, Any]:
    cases = _graph_cases()
    run_frames: list[pd.DataFrame] = []
    graphs: dict[str, ig.Graph] = {}
    for case in cases:
        frame, graph = _run_case(case, seeds=seeds, n_iterations=n_iterations)
        run_frames.append(frame)
        graphs[case.family] = graph
    seed_runs = _claim_columns(pd.concat(run_frames, ignore_index=True, sort=False))
    endpoints = _endpoint_summary(seed_runs, seeds=seeds)
    endpoint_manifest = _frozen_endpoint_manifest(endpoints)
    families = _family_summary(endpoints)
    discovery_curve = _discovery_curve(
        seed_runs,
        endpoint_manifest,
        seeds=seeds,
        permutations=discovery_permutations,
        rng_seed=discovery_rng_seed,
    )
    gates = _gate_matrix(families, discovery_curve)

    recurrent_multi = families[families["recurrent_endpoint_count"].ge(2)]
    top_quality_multi = families[families["top_quality_endpoint_count"].ge(2)]
    summary = {
        "seed_count_per_family": int(seeds),
        "family_count": int(families["family"].nunique()),
        "total_seed_run_rows": int(len(seed_runs)),
        "total_distinct_endpoints": int(endpoints["endpoint_signature_id"].nunique()),
        "frozen_endpoint_manifest_rows": int(len(endpoint_manifest)),
        "discovery_curve_rows": int(len(discovery_curve)),
        "discovery_permutations": int(discovery_permutations),
        "discovery_rng_seed": int(discovery_rng_seed),
        "families_with_recurrent_multi_endpoint": int(len(recurrent_multi)),
        "families_with_multiple_top_quality_endpoints": int(len(top_quality_multi)),
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gates["status"].value_counts().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(seed_runs, output_dir / SEED_RUNS_CSV)
    _write_csv(endpoints, output_dir / ENDPOINT_SUMMARY_CSV)
    _write_csv(endpoint_manifest, output_dir / FROZEN_ENDPOINT_MANIFEST_CSV)
    _write_csv(discovery_curve, output_dir / DISCOVERY_CURVE_CSV)
    _write_csv(families, output_dir / FAMILY_SUMMARY_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    _write_graph_sidecars(graphs, output_dir)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "output_dir": _rel(output_dir),
        "seeds": int(seeds),
        "n_iterations": int(n_iterations),
        "discovery_permutations": int(discovery_permutations),
        "discovery_rng_seed": int(discovery_rng_seed),
        "cases": [
            {
                "family": case.family,
                "mechanism": case.mechanism,
                "gamma": case.gamma,
                "expected_behavior": case.expected_behavior,
            }
            for case in cases
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        family_summary=families,
        endpoint_summary=endpoints,
        endpoint_manifest=endpoint_manifest,
        discovery_curve=discovery_curve,
        gate_matrix=gates,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--n-iterations", type=int, default=-1)
    parser.add_argument("--discovery-permutations", type=int, default=200)
    parser.add_argument("--discovery-rng-seed", type=int, default=1729)
    args = parser.parse_args()
    summary = run_demo(
        output_dir=args.output_dir,
        seeds=args.seeds,
        n_iterations=args.n_iterations,
        discovery_permutations=args.discovery_permutations,
        discovery_rng_seed=args.discovery_rng_seed,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
