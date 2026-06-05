#!/usr/bin/env python3
"""Run a local Leiden+CPM ablation gate for frozen variable-pair panels.

This consumes the frozen counterfactual panel from
``design_leiden_basin_nanoclustering_symmetric_object_variable_pair_counterfactual_panel.py``.
For each selected pair it builds a small induced graph from the pair plus
recoverable top common-neighbor bridge nodes, then runs ordinary Leiden+CPM
under four graph variants:

- original local graph;
- direct pair edge removed;
- pair-to-bridge edges removed;
- direct pair edge and pair-to-bridge edges removed.

This is a controlled local mechanism diagnostic. It does not run the full
NanoClustering graph, execute a route/pathway, promote a wall, inspect
quality/cost as a success claim, or claim method/algorithm success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd

from analyze_leiden_basin_nanoclustering_symmetric_object_variable_pair_graph_mechanisms import (
    _edge_sidecars_for_branch,
    _load_graph_rows,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from sciscape.clustering.runner import LeidenRunner


DEFAULT_PANEL_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_counterfactual_panel_gamma1e5_20260603"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation_gamma1e5_20260603"
)

INPUT_PANEL_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_counterfactual_panel_rows.csv"
)
INPUT_PANEL_CONFIG_JSON = (
    "nanoclustering_symmetric_object_variable_pair_counterfactual_panel_config.json"
)
INPUT_NODE_ROWS_CSV = "nanoclustering_symmetric_object_terminal_difference_node_rows.csv"

LOCAL_GRAPH_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_local_ablation_graph_rows.csv"
)
SEED_RUNS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_local_ablation_seed_runs.csv"
)
VARIANT_SUMMARY_CSV = (
    "nanoclustering_symmetric_object_variable_pair_local_ablation_variant_summary.csv"
)
PAIR_GATE_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_local_ablation_pair_gate_rows.csv"
)
SUMMARY_JSON = (
    "nanoclustering_symmetric_object_variable_pair_local_ablation_summary.json"
)
CONFIG_JSON = "nanoclustering_symmetric_object_variable_pair_local_ablation_config.json"
REPORT_MD = "nanoclustering_symmetric_object_variable_pair_local_ablation_report.md"

RUN_STATUS = "executed_symmetric_object_variable_pair_local_ablation"
CLAIM_BOUNDARY = (
    "NanoClustering symmetric-object variable-pair local ablation only; builds "
    "small induced graphs from a frozen variable-pair panel and runs ordinary "
    "Leiden+CPM under local edge-removal variants. It does not run the full "
    "NanoClustering graph, execute routes/pathways, promote walls, inspect "
    "basin-quality or cost as success claims, or claim method/algorithm success."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_local_ablation_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "plain_leiden_cpm_local_diagnostic_not_method"

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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["run_status"] = RUN_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _parse_top_common_neighbors(value: Any, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return rows
    for rank, item in enumerate(str(value).split(";"), start=1):
        if not item or len(rows) >= int(limit):
            continue
        parts = item.split(":")
        if len(parts) != 3:
            continue
        rows.append(
            {
                "bridge_rank": int(rank),
                "node_id": int(parts[0]),
                "node_scope": str(parts[1]),
                "min_pair_edge_weight": float(parts[2]),
            }
        )
    return rows


def _load_panel_context(panel_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    panel_config = _read_json(panel_dir / INPUT_PANEL_CONFIG_JSON)
    graph_mechanism_dir = Path(str(panel_config["graph_mechanism_dir"]))
    graph_config = _read_json(
        graph_mechanism_dir / "nanoclustering_symmetric_object_variable_pair_graph_mechanism_config.json"
    )
    difference_dir = Path(str(graph_config["difference_dir"]))
    panel = _read_csv(panel_dir / INPUT_PANEL_ROWS_CSV)
    node_rows = _read_csv(difference_dir / INPUT_NODE_ROWS_CSV)
    return panel, node_rows, graph_mechanism_dir, difference_dir


def _node_doc_lookup(node_rows: pd.DataFrame) -> dict[tuple[str, int], int]:
    lookup: dict[tuple[str, int], int] = {}
    for row in node_rows.itertuples(index=False):
        doc_count = getattr(row, "doc_count")
        if pd.isna(doc_count):
            continue
        lookup[(str(row.object_role_universe_id), int(row.node_id))] = max(
            1,
            int(round(float(doc_count))),
        )
    return lookup


def _panel_local_specs(
    *,
    panel: pd.DataFrame,
    doc_lookup: dict[tuple[str, int], int],
    top_common_neighbors: int,
    include_outside_bridges: bool,
) -> tuple[pd.DataFrame, dict[str, set[int]]]:
    rows: list[dict[str, Any]] = []
    target_nodes_by_branch: dict[str, set[int]] = {}
    for index, panel_row in enumerate(panel.itertuples(index=False), start=1):
        data = panel_row._asdict()
        object_role_id = str(data["object_role_universe_id"])
        branch = str(data["branch"])
        left = int(data["left_node_id"])
        right = int(data["right_node_id"])
        bridge_rows = _parse_top_common_neighbors(
            data.get("top_common_neighbors", ""),
            limit=int(top_common_neighbors),
        )
        selected_bridge_rows = []
        excluded_missing = 0
        excluded_outside = 0
        for bridge in bridge_rows:
            node_id = int(bridge["node_id"])
            has_doc = (object_role_id, node_id) in doc_lookup
            if not has_doc and not bool(include_outside_bridges):
                if bridge["node_scope"] == "outside":
                    excluded_outside += 1
                else:
                    excluded_missing += 1
                continue
            selected_bridge_rows.append({**bridge, "has_doc_count": bool(has_doc)})
        node_ids = [left, right]
        for bridge in selected_bridge_rows:
            if int(bridge["node_id"]) not in node_ids:
                node_ids.append(int(bridge["node_id"]))
        target_nodes_by_branch.setdefault(branch, set()).update(node_ids)
        doc_missing = [
            node_id for node_id in node_ids if (object_role_id, int(node_id)) not in doc_lookup
        ]
        rows.append(
            {
                **data,
                "local_pair_id": f"local_pair_{index:03d}",
                "selected_bridge_node_ids": ";".join(
                    str(int(bridge["node_id"])) for bridge in selected_bridge_rows
                ),
                "selected_bridge_rank_scope_weight": ";".join(
                    (
                        f"{int(bridge['bridge_rank'])}:"
                        f"{int(bridge['node_id'])}:"
                        f"{bridge['node_scope']}:"
                        f"{float(bridge['min_pair_edge_weight'])}"
                    )
                    for bridge in selected_bridge_rows
                ),
                "selected_bridge_count": int(len(selected_bridge_rows)),
                "excluded_outside_bridge_count": int(excluded_outside),
                "excluded_missing_doc_bridge_count": int(excluded_missing),
                "local_node_ids": ";".join(str(int(node_id)) for node_id in node_ids),
                "local_node_count": int(len(node_ids)),
                "local_doc_missing_node_count": int(len(doc_missing)),
                "local_doc_missing_node_ids": ";".join(str(int(node_id)) for node_id in doc_missing),
            }
        )
    return pd.DataFrame(rows), target_nodes_by_branch


def _collect_induced_edges_by_branch(
    *,
    graph_mechanism_dir: Path,
    target_nodes_by_branch: dict[str, set[int]],
    edge_chunk_size: int,
) -> dict[str, pd.DataFrame]:
    graph_config = _read_json(
        graph_mechanism_dir / "nanoclustering_symmetric_object_variable_pair_graph_mechanism_config.json"
    )
    graph_rows = _load_graph_rows(Path(str(graph_config["difference_dir"])))
    induced: dict[str, pd.DataFrame] = {}
    for branch, target_nodes in sorted(target_nodes_by_branch.items()):
        targets = np.asarray(sorted(target_nodes), dtype=np.uint32)
        if targets.size == 0:
            induced[branch] = pd.DataFrame(columns=["source", "target", "weight"])
            continue
        src_path, dst_path, weight_path = _edge_sidecars_for_branch(graph_rows, str(branch))
        edge_src = np.memmap(src_path, dtype=np.uint32, mode="r")
        edge_dst = np.memmap(dst_path, dtype=np.uint32, mode="r")
        edge_weight = np.memmap(weight_path, dtype=np.float64, mode="r")
        parts: list[pd.DataFrame] = []
        n_edges = int(edge_weight.shape[0])
        for start in range(0, n_edges, int(edge_chunk_size)):
            stop = min(start + int(edge_chunk_size), n_edges)
            src = np.asarray(edge_src[start:stop], dtype=np.uint32)
            dst = np.asarray(edge_dst[start:stop], dtype=np.uint32)
            hit = np.isin(src, targets) & np.isin(dst, targets)
            if not bool(hit.any()):
                continue
            left = src[hit].astype(np.uint32)
            right = dst[hit].astype(np.uint32)
            lo = np.minimum(left, right)
            hi = np.maximum(left, right)
            parts.append(
                pd.DataFrame(
                    {
                        "source": lo.astype(np.uint32),
                        "target": hi.astype(np.uint32),
                        "weight": np.asarray(edge_weight[start:stop], dtype=np.float64)[hit],
                    }
                )
            )
        if not parts:
            induced[branch] = pd.DataFrame(columns=["source", "target", "weight"])
            continue
        frame = pd.concat(parts, ignore_index=True)
        frame = (
            frame.groupby(["source", "target"], as_index=False, sort=True)["weight"]
            .sum()
            .sort_values(["source", "target"], kind="mergesort")
        )
        induced[branch] = frame
    return induced


def _local_edges_for_spec(
    *,
    induced_edges: pd.DataFrame,
    node_ids: list[int],
    left_node: int,
    right_node: int,
    bridge_nodes: set[int],
    graph_variant: str,
) -> pd.DataFrame:
    node_set = set(int(node_id) for node_id in node_ids)
    edges = induced_edges[
        induced_edges["source"].astype(int).isin(node_set)
        & induced_edges["target"].astype(int).isin(node_set)
    ].copy()
    if edges.empty:
        return edges
    direct_key = tuple(sorted((int(left_node), int(right_node))))

    def remove_row(row: pd.Series) -> bool:
        source = int(row["source"])
        target = int(row["target"])
        key = tuple(sorted((source, target)))
        is_direct = key == direct_key
        is_pair_bridge = (
            (source in {left_node, right_node} and target in bridge_nodes)
            or (target in {left_node, right_node} and source in bridge_nodes)
        )
        if graph_variant == "original":
            return False
        if graph_variant == "drop_direct_edge":
            return is_direct
        if graph_variant == "drop_bridge_edges":
            return is_pair_bridge
        if graph_variant == "drop_direct_and_bridge_edges":
            return is_direct or is_pair_bridge
        raise ValueError(f"unknown graph variant: {graph_variant}")

    keep = ~edges.apply(remove_row, axis=1)
    return edges[keep].copy()


def _build_igraph(node_ids: list[int], edges: pd.DataFrame) -> ig.Graph:
    names = [str(int(node_id)) for node_id in node_ids]
    index = {int(node_id): offset for offset, node_id in enumerate(node_ids)}
    if edges.empty:
        graph = ig.Graph(n=len(node_ids), directed=False)
        graph.vs["name"] = names
        graph.es["weight"] = []
        return graph
    graph = ig.Graph(
        n=len(node_ids),
        edges=[
            (index[int(row.source)], index[int(row.target)])
            for row in edges.itertuples(index=False)
            if int(row.source) != int(row.target)
        ],
        directed=False,
    )
    graph.vs["name"] = names
    graph.es["weight"] = [
        float(row.weight)
        for row in edges.itertuples(index=False)
        if int(row.source) != int(row.target)
    ]
    return graph


def _initial_membership(
    *,
    start_condition: str,
    node_ids: list[int],
    left_node: int,
    right_node: int,
    bridge_nodes: set[int],
) -> list[int]:
    index = {int(node_id): offset for offset, node_id in enumerate(node_ids)}
    membership = list(range(len(node_ids)))
    if start_condition == "singleton":
        return membership
    if start_condition == "pair_together":
        membership[index[int(right_node)]] = membership[index[int(left_node)]]
        return _renumber(membership)
    if start_condition == "bridges_to_left":
        label = membership[index[int(left_node)]]
        for node_id in bridge_nodes:
            if int(node_id) in index:
                membership[index[int(node_id)]] = label
        return _renumber(membership)
    if start_condition == "bridges_to_right":
        label = membership[index[int(right_node)]]
        for node_id in bridge_nodes:
            if int(node_id) in index:
                membership[index[int(node_id)]] = label
        return _renumber(membership)
    if start_condition == "all_local_together":
        return [0 for _ in node_ids]
    raise ValueError(f"unknown start condition: {start_condition}")


def _renumber(membership: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    next_label = 0
    output: list[int] = []
    for label in membership:
        label = int(label)
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        output.append(mapping[label])
    return output


def _canonical_groups(node_ids: list[int], membership: list[int]) -> list[list[str]]:
    groups: dict[int, list[str]] = {}
    for node_id, label in zip(node_ids, membership, strict=True):
        groups.setdefault(int(label), []).append(str(int(node_id)))
    return sorted(sorted(nodes, key=lambda value: int(value)) for nodes in groups.values())


def _signature_id(groups: list[list[str]]) -> str:
    payload = json.dumps(groups, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _mechanism_read(
    *,
    membership: list[int],
    node_ids: list[int],
    left_node: int,
    right_node: int,
    bridge_nodes: set[int],
) -> dict[str, Any]:
    labels = {int(node_id): int(label) for node_id, label in zip(node_ids, membership, strict=True)}
    left_label = labels[int(left_node)]
    right_label = labels[int(right_node)]
    pair_coassigned = left_label == right_label
    left_bridge_count = sum(1 for node_id in bridge_nodes if labels.get(int(node_id)) == left_label)
    right_bridge_count = sum(1 for node_id in bridge_nodes if labels.get(int(node_id)) == right_label)
    pair_bridge_count = left_bridge_count if pair_coassigned else 0
    if pair_coassigned and pair_bridge_count > 0:
        read = "pair_coassigned_with_selected_bridge"
    elif pair_coassigned:
        read = "pair_coassigned_without_selected_bridge"
    elif left_bridge_count > 0 and right_bridge_count > 0:
        read = "pair_separated_bridge_split"
    elif left_bridge_count > 0 or right_bridge_count > 0:
        read = "pair_separated_single_side_bridge"
    else:
        read = "pair_separated_no_selected_bridge"
    return {
        "pair_coassigned": bool(pair_coassigned),
        "left_bridge_same_cluster_count": int(left_bridge_count),
        "right_bridge_same_cluster_count": int(right_bridge_count),
        "pair_bridge_same_cluster_count": int(pair_bridge_count),
        "mechanism_read": read,
    }


def _run_local_ablations(
    *,
    specs: pd.DataFrame,
    induced_edges_by_branch: dict[str, pd.DataFrame],
    doc_lookup: dict[tuple[str, int], int],
    gamma: float,
    seeds: int,
    n_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for spec in specs.itertuples(index=False):
        data = spec._asdict()
        object_role_id = str(data["object_role_universe_id"])
        branch = str(data["branch"])
        local_pair_id = str(data["local_pair_id"])
        left = int(data["left_node_id"])
        right = int(data["right_node_id"])
        node_ids = [int(value) for value in str(data["local_node_ids"]).split(";") if value]
        bridge_nodes = {
            int(value)
            for value in str(data["selected_bridge_node_ids"]).split(";")
            if value
        }
        node_sizes = [
            int(doc_lookup.get((object_role_id, int(node_id)), 1))
            for node_id in node_ids
        ]
        induced_edges = induced_edges_by_branch.get(
            branch,
            pd.DataFrame(columns=["source", "target", "weight"]),
        )
        for graph_variant in GRAPH_VARIANTS:
            local_edges = _local_edges_for_spec(
                induced_edges=induced_edges,
                node_ids=node_ids,
                left_node=left,
                right_node=right,
                bridge_nodes=bridge_nodes,
                graph_variant=graph_variant,
            )
            graph = _build_igraph(node_ids, local_edges)
            removed_edge_weight = float(
                _local_edges_for_spec(
                    induced_edges=induced_edges,
                    node_ids=node_ids,
                    left_node=left,
                    right_node=right,
                    bridge_nodes=bridge_nodes,
                    graph_variant="original",
                )["weight"].sum()
                - local_edges["weight"].sum()
            )
            graph_rows.append(
                {
                    **_spec_identity(data),
                    "graph_variant": graph_variant,
                    "local_node_count": int(len(node_ids)),
                    "selected_bridge_count": int(len(bridge_nodes)),
                    "local_edge_count": int(graph.ecount()),
                    "local_edge_weight_sum": float(sum(graph.es["weight"])) if graph.ecount() else 0.0,
                    "removed_edge_weight": float(removed_edge_weight),
                    "node_size_sum": int(sum(node_sizes)),
                    "node_size_min": int(min(node_sizes)) if node_sizes else None,
                    "node_size_max": int(max(node_sizes)) if node_sizes else None,
                }
            )
            runner = LeidenRunner(graph, objective="cpm", default_iterations=int(n_iterations))
            for start_condition in START_CONDITIONS:
                initial = _initial_membership(
                    start_condition=start_condition,
                    node_ids=node_ids,
                    left_node=left,
                    right_node=right,
                    bridge_nodes=bridge_nodes,
                )
                for seed in range(int(seeds)):
                    result = runner.run(
                        float(gamma),
                        seed=int(seed),
                        initial_membership=initial,
                        node_sizes=node_sizes,
                    )
                    membership = list(map(int, result.membership))
                    groups = _canonical_groups(node_ids, membership)
                    read = _mechanism_read(
                        membership=membership,
                        node_ids=node_ids,
                        left_node=left,
                        right_node=right,
                        bridge_nodes=bridge_nodes,
                    )
                    run_rows.append(
                        {
                            **_spec_identity(data),
                            "graph_variant": graph_variant,
                            "start_condition": start_condition,
                            "seed": int(seed),
                            "gamma": float(gamma),
                            "n_iterations": int(n_iterations),
                            "local_node_count": int(len(node_ids)),
                            "selected_bridge_count": int(len(bridge_nodes)),
                            "local_edge_count": int(graph.ecount()),
                            "cluster_count": int(result.cluster_count),
                            "quality": float(result.quality),
                            "endpoint_signature_id": _signature_id(groups),
                            "endpoint_signature": json.dumps(groups, sort_keys=True),
                            **read,
                        }
                    )
    return _claim_columns(pd.DataFrame(graph_rows)), _claim_columns(pd.DataFrame(run_rows))


def _spec_identity(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "local_pair_id": str(data["local_pair_id"]),
        "object_role_universe_id": str(data["object_role_universe_id"]),
        "branch": str(data["branch"]),
        "left_node_id": int(data["left_node_id"]),
        "right_node_id": int(data["right_node_id"]),
        "pair_scope": str(data["pair_scope"]),
        "counterfactual_class": str(data["counterfactual_class"]),
        "selection_reason": str(data["selection_reason"]),
    }


def _variant_summary(seed_runs: pd.DataFrame) -> pd.DataFrame:
    if seed_runs.empty:
        return _claim_columns(pd.DataFrame())
    rows: list[dict[str, Any]] = []
    group_cols = [
        "local_pair_id",
        "object_role_universe_id",
        "branch",
        "left_node_id",
        "right_node_id",
        "pair_scope",
        "counterfactual_class",
        "selection_reason",
        "graph_variant",
    ]
    recurrent_floor = lambda count: max(2, int(math.ceil(count * 0.05)))
    for keys, group in seed_runs.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        endpoint_counts = group["endpoint_signature_id"].value_counts()
        floor = recurrent_floor(int(len(group)))
        recurrent = endpoint_counts[endpoint_counts.ge(floor)]
        pair_coassigned = group["pair_coassigned"].astype(bool)
        read_counts = group["mechanism_read"].value_counts().to_dict()
        rows.append(
            {
                **key_data,
                "run_count": int(len(group)),
                "start_condition_count": int(group["start_condition"].nunique()),
                "seed_count": int(group["seed"].nunique()),
                "local_node_count": int(group["local_node_count"].min()),
                "selected_bridge_count": int(group["selected_bridge_count"].min()),
                "local_edge_count": int(group["local_edge_count"].min()),
                "distinct_endpoint_count": int(endpoint_counts.shape[0]),
                "recurrent_endpoint_count": int(recurrent.shape[0]),
                "recurrent_endpoint_floor": int(floor),
                "top_endpoint_seed_count": int(endpoint_counts.max()),
                "top_endpoint_share": float(endpoint_counts.max() / len(group)),
                "pair_coassigned_run_count": int(pair_coassigned.sum()),
                "pair_coassigned_share": float(pair_coassigned.mean()),
                "pair_bridge_same_cluster_median": float(
                    group["pair_bridge_same_cluster_count"].median()
                ),
                "quality_min": float(group["quality"].min()),
                "quality_median": float(group["quality"].median()),
                "quality_max": float(group["quality"].max()),
                "cluster_count_min": int(group["cluster_count"].min()),
                "cluster_count_median": float(group["cluster_count"].median()),
                "cluster_count_max": int(group["cluster_count"].max()),
                "mechanism_read_counts": json.dumps(read_counts, sort_keys=True),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _pair_gate_rows(
    *,
    variant_summary: pd.DataFrame,
    effect_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if variant_summary.empty:
        return _claim_columns(pd.DataFrame())
    group_cols = [
        "local_pair_id",
        "object_role_universe_id",
        "branch",
        "left_node_id",
        "right_node_id",
        "pair_scope",
        "counterfactual_class",
        "selection_reason",
    ]
    for keys, group in variant_summary.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        by_variant = group.set_index("graph_variant")

        def metric(variant: str, column: str, default: float | int | None = None):
            if variant not in by_variant.index:
                return default
            return by_variant.loc[variant, column]

        original_share = float(metric("original", "pair_coassigned_share", 0.0))
        drop_direct_share = float(metric("drop_direct_edge", "pair_coassigned_share", 0.0))
        drop_bridge_share = float(metric("drop_bridge_edges", "pair_coassigned_share", 0.0))
        drop_both_share = float(
            metric("drop_direct_and_bridge_edges", "pair_coassigned_share", 0.0)
        )
        direct_delta = drop_direct_share - original_share
        bridge_delta = drop_bridge_share - original_share
        both_delta = drop_both_share - original_share
        original_distinct = int(metric("original", "distinct_endpoint_count", 0))
        original_recurrent = int(metric("original", "recurrent_endpoint_count", 0))
        original_switch = original_distinct > 1 or (0.0 < original_share < 1.0)

        if original_share <= 0.0:
            gate_class = "not_reproduced_no_original_local_coassignment"
        elif direct_delta <= -float(effect_threshold) and bridge_delta <= -float(effect_threshold):
            gate_class = "direct_and_bridge_sensitive_local_switch"
        elif direct_delta <= -float(effect_threshold):
            gate_class = "direct_edge_sensitive_local_switch"
        elif bridge_delta <= -float(effect_threshold):
            gate_class = "bridge_sensitive_local_switch"
        elif original_switch:
            gate_class = "local_seed_or_start_sensitive_switch"
        elif min(drop_direct_share, drop_bridge_share, drop_both_share) >= 0.9:
            gate_class = "robust_local_collapse_not_mechanism_specific"
        else:
            gate_class = "weak_or_ambiguous_local_ablation_effect"

        gate_status = (
            "diagnostic_supports_local_mechanism_reproduction"
            if gate_class
            in {
                "direct_and_bridge_sensitive_local_switch",
                "direct_edge_sensitive_local_switch",
                "bridge_sensitive_local_switch",
                "local_seed_or_start_sensitive_switch",
            }
            else "diagnostic_only_no_local_mechanism_reproduction"
        )
        rows.append(
            {
                **key_data,
                "original_pair_coassigned_share": original_share,
                "drop_direct_pair_coassigned_share": drop_direct_share,
                "drop_bridge_pair_coassigned_share": drop_bridge_share,
                "drop_direct_and_bridge_pair_coassigned_share": drop_both_share,
                "drop_direct_pair_coassigned_delta": direct_delta,
                "drop_bridge_pair_coassigned_delta": bridge_delta,
                "drop_direct_and_bridge_pair_coassigned_delta": both_delta,
                "original_distinct_endpoint_count": original_distinct,
                "original_recurrent_endpoint_count": original_recurrent,
                "original_has_local_switch_signal": bool(original_switch),
                "gate_class": gate_class,
                "gate_status": gate_status,
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _build_summary(
    *,
    output_dir: Path,
    panel_dir: Path,
    graph_rows: pd.DataFrame,
    seed_runs: pd.DataFrame,
    variant_summary: pd.DataFrame,
    pair_gate_rows: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_symmetric_object_variable_pair_local_ablation_summary.v1",
        "status": RUN_STATUS,
        "panel_dir": str(panel_dir),
        "output_dir": str(output_dir),
        "pair_count": int(pair_gate_rows["local_pair_id"].nunique()) if not pair_gate_rows.empty else 0,
        "graph_variant_count": int(graph_rows["graph_variant"].nunique()) if not graph_rows.empty else 0,
        "seed_run_count": int(len(seed_runs)),
        "variant_summary_count": int(len(variant_summary)),
        "gate_class_counts": pair_gate_rows["gate_class"].value_counts().to_dict()
        if not pair_gate_rows.empty
        else {},
        "gate_status_counts": pair_gate_rows["gate_status"].value_counts().to_dict()
        if not pair_gate_rows.empty
        else {},
        "object_count": int(pair_gate_rows["object_role_universe_id"].nunique())
        if not pair_gate_rows.empty
        else 0,
        "pair_scope_counts": pair_gate_rows["pair_scope"].value_counts().to_dict()
        if not pair_gate_rows.empty
        else {},
        "counterfactual_class_counts": pair_gate_rows[
            "counterfactual_class"
        ].value_counts().to_dict()
        if not pair_gate_rows.empty
        else {},
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_gate_rows: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Symmetric-Object Variable-Pair Local Ablation",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_count: {summary['pair_count']}",
        f"- graph_variant_count: {summary['graph_variant_count']}",
        f"- seed_run_count: {summary['seed_run_count']}",
        f"- gate_class_counts: {summary['gate_class_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- pair_scope_counts: {summary['pair_scope_counts']}",
        f"- counterfactual_class_counts: {summary['counterfactual_class_counts']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Gates",
    ]
    if pair_gate_rows.empty:
        lines.append("- no rows")
    else:
        for row in pair_gate_rows.sort_values(
            ["gate_status", "gate_class", "object_role_universe_id", "local_pair_id"],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).itertuples(index=False):
            lines.append(
                "- "
                f"{row.local_pair_id} {row.object_role_universe_id} "
                f"{row.left_node_id}-{row.right_node_id} "
                f"class={row.gate_class} status={row.gate_status} "
                f"orig={row.original_pair_coassigned_share:.3f} "
                f"drop_direct={row.drop_direct_pair_coassigned_share:.3f} "
                f"drop_bridge={row.drop_bridge_pair_coassigned_share:.3f} "
                f"drop_both={row.drop_direct_and_bridge_pair_coassigned_share:.3f} "
                f"orig_endpoints={row.original_distinct_endpoint_count}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This gate tests whether a frozen panel's local induced graph "
                "can reproduce a pair co-assignment switch or ablation-sensitive "
                "mechanism under ordinary Leiden+CPM. It is not a full-graph "
                "NanoClustering replay, route/pathway execution, wall result, "
                "quality/cost result, or method claim."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    panel_dir = Path(args.panel_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel, node_rows, graph_mechanism_dir, difference_dir = _load_panel_context(panel_dir)
    doc_lookup = _node_doc_lookup(node_rows)
    specs, target_nodes_by_branch = _panel_local_specs(
        panel=panel,
        doc_lookup=doc_lookup,
        top_common_neighbors=int(args.top_common_neighbors),
        include_outside_bridges=bool(args.include_outside_bridges),
    )
    induced_edges_by_branch = _collect_induced_edges_by_branch(
        graph_mechanism_dir=graph_mechanism_dir,
        target_nodes_by_branch=target_nodes_by_branch,
        edge_chunk_size=int(args.edge_chunk_size),
    )
    graph_rows, seed_runs = _run_local_ablations(
        specs=specs,
        induced_edges_by_branch=induced_edges_by_branch,
        doc_lookup=doc_lookup,
        gamma=float(args.gamma),
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
    )
    variant_summary = _variant_summary(seed_runs)
    pair_gate_rows = _pair_gate_rows(
        variant_summary=variant_summary,
        effect_threshold=float(args.effect_threshold),
    )
    _write_csv(_claim_columns(specs), output_dir / LOCAL_GRAPH_ROWS_CSV)
    _write_csv(graph_rows, output_dir / LOCAL_GRAPH_ROWS_CSV.replace("_graph_rows", "_variant_graph_rows"))
    _write_csv(seed_runs, output_dir / SEED_RUNS_CSV)
    _write_csv(variant_summary, output_dir / VARIANT_SUMMARY_CSV)
    _write_csv(pair_gate_rows, output_dir / PAIR_GATE_ROWS_CSV)
    summary = _build_summary(
        output_dir=output_dir,
        panel_dir=panel_dir,
        graph_rows=graph_rows,
        seed_runs=seed_runs,
        variant_summary=variant_summary,
        pair_gate_rows=pair_gate_rows,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_symmetric_object_variable_pair_local_ablation.v1",
        "panel_dir": str(panel_dir),
        "graph_mechanism_dir": str(graph_mechanism_dir),
        "difference_dir": str(difference_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "top_common_neighbors": int(args.top_common_neighbors),
        "include_outside_bridges": bool(args.include_outside_bridges),
        "edge_chunk_size": int(args.edge_chunk_size),
        "effect_threshold": float(args.effect_threshold),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, pair_gate_rows=pair_gate_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gamma", type=float, default=1.0e-5)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--top-common-neighbors", type=int, default=10)
    parser.add_argument("--include-outside-bridges", action="store_true")
    parser.add_argument("--edge-chunk-size", type=int, default=5_000_000)
    parser.add_argument("--effect-threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
