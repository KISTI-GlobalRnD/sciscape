#!/usr/bin/env python3
"""Score variable terminal node-pairs against graph-local mechanisms.

This reads the variable node-pairs emitted by
``analyze_leiden_basin_nanoclustering_symmetric_object_terminal_membership_differences.py``
and joins graph-local evidence:

- direct edge weight between the variable pair;
- doc-weighted CPM pair delta and critical gamma;
- shared-neighbor bridge mass around the pair;
- object/support/outside scope of common neighbors.

It is a mechanism diagnostic only. It does not run Leiden, promote wall/pathway,
inspect basin quality/cost as a success claim, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sciscape.clustering.integer_remap import ensure_int_edge_sidecars

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    GRAPH_INPUT_ROWS_CSV,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_DIFFERENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_terminal_membership_difference_review_gamma1e5_20260603"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_graph_mechanisms_gamma1e5_20260603"
)

INPUT_NODE_PAIR_ROWS_CSV = (
    "nanoclustering_symmetric_object_terminal_difference_variable_node_pair_rows.csv"
)
INPUT_NODE_ROWS_CSV = "nanoclustering_symmetric_object_terminal_difference_node_rows.csv"
INPUT_OBJECT_ROWS_CSV = "nanoclustering_symmetric_object_terminal_difference_object_rows.csv"
PAIR_ROWS_CSV = "nanoclustering_symmetric_object_variable_pair_graph_mechanism_rows.csv"
OBJECT_ROWS_CSV = "nanoclustering_symmetric_object_variable_pair_graph_mechanism_object_rows.csv"
SUMMARY_JSON = "nanoclustering_symmetric_object_variable_pair_graph_mechanism_summary.json"
REPORT_MD = "nanoclustering_symmetric_object_variable_pair_graph_mechanism_report.md"
CONFIG_JSON = "nanoclustering_symmetric_object_variable_pair_graph_mechanism_config.json"

RUN_STATUS = "executed_symmetric_object_variable_pair_graph_mechanism_review"
CLAIM_BOUNDARY = (
    "NanoClustering symmetric-object variable-pair graph mechanism review only; "
    "scores saved variable terminal node-pairs by direct edge, doc-weighted CPM "
    "delta, critical gamma, and shared-neighbor bridge mass. It does not run "
    "Leiden, promote wall/pathway, basin-quality, cost, real-data method-success, "
    "or algorithm claims."
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return _read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _prefix_stats(prefix: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_p90": None,
        }
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
    }


def _load_graph_rows(difference_dir: Path) -> pd.DataFrame:
    difference_config = _read_json(
        difference_dir / "nanoclustering_symmetric_object_terminal_difference_config.json"
    )
    multistart_dir = Path(str(difference_config["multistart_dir"]))
    if not multistart_dir.is_absolute():
        multistart_dir = Path.cwd() / multistart_dir
    multistart_config = _read_json(
        multistart_dir / "nanoclustering_symmetric_object_multistart_config.json"
    )
    readiness_dir = Path(str(multistart_config["readiness_dir"]))
    return _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)


def _edge_sidecars_for_branch(graph_rows: pd.DataFrame, branch: str) -> tuple[Path, Path, Path]:
    rows = graph_rows[graph_rows["branch"].astype(str).eq(str(branch))]
    if rows.empty:
        raise ValueError(f"missing graph row for branch: {branch}")
    src_path, dst_path, weight_path = ensure_int_edge_sidecars(
        Path(str(rows.iloc[0]["runtime_int_edges_path"]))
    )
    return src_path, dst_path, weight_path


def _target_adjacency(
    *,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_weight: np.ndarray,
    target_nodes: np.ndarray,
    chunk_size: int,
) -> tuple[dict[int, dict[int, float]], dict[int, float]]:
    target_nodes = np.asarray(sorted(set(int(node) for node in target_nodes)), dtype=np.uint32)
    adjacency: dict[int, dict[int, float]] = {int(node): {} for node in target_nodes}
    weighted_degree: dict[int, float] = {int(node): 0.0 for node in target_nodes}
    n_edges = int(edge_weight.shape[0])
    for start in range(0, n_edges, int(chunk_size)):
        stop = min(start + int(chunk_size), n_edges)
        src = np.asarray(edge_src[start:stop], dtype=np.uint32)
        dst = np.asarray(edge_dst[start:stop], dtype=np.uint32)
        weights = np.asarray(edge_weight[start:stop], dtype=np.float64)
        src_hit = np.isin(src, target_nodes)
        dst_hit = np.isin(dst, target_nodes)
        if bool(src_hit.any()):
            for s, d, weight in zip(src[src_hit], dst[src_hit], weights[src_hit], strict=True):
                si = int(s)
                di = int(d)
                w = float(weight)
                adjacency[si][di] = adjacency[si].get(di, 0.0) + w
                weighted_degree[si] += w
        if bool(dst_hit.any()):
            for s, d, weight in zip(src[dst_hit], dst[dst_hit], weights[dst_hit], strict=True):
                si = int(s)
                di = int(d)
                if si == di:
                    continue
                w = float(weight)
                adjacency[di][si] = adjacency[di].get(si, 0.0) + w
                weighted_degree[di] += w
    return adjacency, weighted_degree


def _scope_for_common_neighbor(node: int, universe_scope: dict[int, str]) -> str:
    return universe_scope.get(int(node), "outside")


def _common_neighbor_stats(
    *,
    left_node: int,
    right_node: int,
    left_neighbors: dict[int, float],
    right_neighbors: dict[int, float],
    universe_scope: dict[int, str],
) -> dict[str, Any]:
    common = (set(left_neighbors) & set(right_neighbors)) - {int(left_node), int(right_node)}
    if not common:
        return {
            "common_neighbor_count": 0,
            "common_neighbor_min_weight_sum": 0.0,
            "common_neighbor_sum_weight_sum": 0.0,
            "common_neighbor_object_count": 0,
            "common_neighbor_support_count": 0,
            "common_neighbor_outside_count": 0,
            "common_neighbor_object_min_weight_sum": 0.0,
            "common_neighbor_support_min_weight_sum": 0.0,
            "common_neighbor_outside_min_weight_sum": 0.0,
            "top_common_neighbors": "",
        }
    min_sum = 0.0
    sum_sum = 0.0
    scope_counts = {"object": 0, "support": 0, "outside": 0}
    scope_min = {"object": 0.0, "support": 0.0, "outside": 0.0}
    top_rows: list[tuple[float, int, str]] = []
    for node in common:
        left_weight = float(left_neighbors[node])
        right_weight = float(right_neighbors[node])
        min_weight = min(left_weight, right_weight)
        sum_weight = left_weight + right_weight
        scope = _scope_for_common_neighbor(int(node), universe_scope)
        if scope not in scope_counts:
            scope = "outside"
        min_sum += min_weight
        sum_sum += sum_weight
        scope_counts[scope] += 1
        scope_min[scope] += min_weight
        top_rows.append((min_weight, int(node), scope))
    top_rows.sort(reverse=True)
    return {
        "common_neighbor_count": int(len(common)),
        "common_neighbor_min_weight_sum": float(min_sum),
        "common_neighbor_sum_weight_sum": float(sum_sum),
        "common_neighbor_object_count": int(scope_counts["object"]),
        "common_neighbor_support_count": int(scope_counts["support"]),
        "common_neighbor_outside_count": int(scope_counts["outside"]),
        "common_neighbor_object_min_weight_sum": float(scope_min["object"]),
        "common_neighbor_support_min_weight_sum": float(scope_min["support"]),
        "common_neighbor_outside_min_weight_sum": float(scope_min["outside"]),
        "top_common_neighbors": ";".join(
            f"{node}:{scope}:{weight}" for weight, node, scope in top_rows[:10]
        ),
    }


def _pair_mechanism_label(row: dict[str, Any], gamma: float) -> str:
    if float(row["direct_cpm_delta_q"]) > 0.0:
        return "direct_positive_weak_pair_at_gamma"
    if float(row["direct_edge_weight"]) > 0.0 and float(row["direct_critical_gamma"] or 0.0) >= gamma:
        return "near_direct_weak_pair"
    if float(row["common_neighbor_min_weight_sum"]) > 0.0:
        return "shared_neighbor_bridge_mediated_pair"
    return "no_direct_or_shared_graph_mass_detected"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    difference_dir = Path(args.difference_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variable_pairs = _read_csv_or_empty(difference_dir / INPUT_NODE_PAIR_ROWS_CSV)
    node_rows = _read_csv(difference_dir / INPUT_NODE_ROWS_CSV)
    object_rows = _read_csv(difference_dir / INPUT_OBJECT_ROWS_CSV)
    if variable_pairs.empty:
        pair_frame = pd.DataFrame()
        object_frame = _empty_object_frame(object_rows)
    else:
        graph_rows = _load_graph_rows(difference_dir)
        branch_by_object = object_rows.set_index("object_role_universe_id")[
            "branch"
        ].astype(str).to_dict()
        pair_records: list[dict[str, Any]] = []
        for branch, branch_pairs in variable_pairs.assign(
            branch=variable_pairs["object_role_universe_id"].map(branch_by_object)
        ).groupby("branch", sort=False):
            if pd.isna(branch):
                raise ValueError("variable pair has no branch after object join")
            target_nodes = np.unique(
                np.concatenate(
                    [
                        branch_pairs["left_node_id"].to_numpy(dtype=np.uint32),
                        branch_pairs["right_node_id"].to_numpy(dtype=np.uint32),
                    ]
                )
            )
            src_path, dst_path, weight_path = _edge_sidecars_for_branch(graph_rows, str(branch))
            edge_src = np.memmap(src_path, dtype=np.uint32, mode="r")
            edge_dst = np.memmap(dst_path, dtype=np.uint32, mode="r")
            edge_weight = np.memmap(weight_path, dtype=np.float64, mode="r")
            adjacency, weighted_degree = _target_adjacency(
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_weight=edge_weight,
                target_nodes=target_nodes,
                chunk_size=int(args.edge_chunk_size),
            )
            for object_role_id, group in branch_pairs.groupby(
                "object_role_universe_id", sort=False
            ):
                universe_scope = (
                    node_rows[
                        node_rows["object_role_universe_id"].astype(str).eq(
                            str(object_role_id)
                        )
                    ]
                    .set_index("node_id")["node_scope"]
                    .astype(str)
                    .to_dict()
                )
                for pair in group.itertuples(index=False):
                    data = pair._asdict()
                    left_node = int(data["left_node_id"])
                    right_node = int(data["right_node_id"])
                    left_neighbors = adjacency.get(left_node, {})
                    right_neighbors = adjacency.get(right_node, {})
                    direct_weight = float(left_neighbors.get(right_node, 0.0))
                    penalty_factor = float(data["left_doc_count"]) * float(
                        data["right_doc_count"]
                    )
                    penalty = float(args.gamma) * penalty_factor
                    delta = direct_weight - penalty
                    critical_gamma = (
                        float(direct_weight / penalty_factor)
                        if penalty_factor > 0.0
                        else None
                    )
                    row = {
                        **data,
                        "branch": str(branch),
                        "direct_edge_weight": direct_weight,
                        "direct_edge_present": direct_weight > 0.0,
                        "left_weighted_degree": float(weighted_degree.get(left_node, 0.0)),
                        "right_weighted_degree": float(weighted_degree.get(right_node, 0.0)),
                        "left_direct_edge_degree_share": (
                            float(direct_weight / weighted_degree[left_node])
                            if weighted_degree.get(left_node, 0.0) > 0.0
                            else None
                        ),
                        "right_direct_edge_degree_share": (
                            float(direct_weight / weighted_degree[right_node])
                            if weighted_degree.get(right_node, 0.0) > 0.0
                            else None
                        ),
                        "penalty_factor_doc_product": penalty_factor,
                        "direct_cpm_penalty_at_gamma": penalty,
                        "direct_cpm_delta_q": delta,
                        "direct_critical_gamma": critical_gamma,
                        "direct_positive_at_gamma": delta > 0.0,
                        "direct_positive_at_gamma3e5": (
                            direct_weight - (3.0e-5 * penalty_factor)
                        )
                        > 0.0,
                        "direct_positive_at_gamma1e4": (
                            direct_weight - (1.0e-4 * penalty_factor)
                        )
                        > 0.0,
                    }
                    row.update(
                        _common_neighbor_stats(
                            left_node=left_node,
                            right_node=right_node,
                            left_neighbors=left_neighbors,
                            right_neighbors=right_neighbors,
                            universe_scope=universe_scope,
                        )
                    )
                    row["mechanism_label"] = _pair_mechanism_label(
                        row,
                        gamma=float(args.gamma),
                    )
                    row["run_status"] = RUN_STATUS
                    row["claim_boundary"] = CLAIM_BOUNDARY
                    pair_records.append(row)
        pair_frame = pd.DataFrame(pair_records)
        object_frame = _object_summary(pair_frame=pair_frame, object_rows=object_rows)

    _write_csv(pair_frame, output_dir / PAIR_ROWS_CSV)
    _write_csv(object_frame, output_dir / OBJECT_ROWS_CSV)
    summary = _build_summary(
        difference_dir=difference_dir,
        output_dir=output_dir,
        pair_frame=pair_frame,
        object_frame=object_frame,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_symmetric_object_variable_pair_graph_mechanism_review.v1",
        "difference_dir": str(difference_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "edge_chunk_size": int(args.edge_chunk_size),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, object_frame=object_frame)
    return summary


def _empty_object_frame(object_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in object_rows.itertuples(index=False):
        data = row._asdict()
        rows.append(
            {
                "object_role_universe_id": data["object_role_universe_id"],
                "branch": data["branch"],
                "variable_pair_count": 0,
                "direct_edge_present_pair_count": 0,
                "direct_positive_pair_count": 0,
                "shared_neighbor_pair_count": 0,
                "mechanism_status": "closed_control_no_variable_pairs",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _object_summary(*, pair_frame: pd.DataFrame, object_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    object_meta = object_rows.set_index("object_role_universe_id").to_dict("index")
    for object_role_id, group in pair_frame.groupby("object_role_universe_id", sort=False):
        direct_present = group["direct_edge_present"].astype(bool)
        direct_positive = group["direct_positive_at_gamma"].astype(bool)
        shared_neighbor = group["common_neighbor_count"].astype(int) > 0
        labels = group["mechanism_label"].value_counts().to_dict()
        if bool(direct_positive.any()):
            mechanism_status = "has_direct_positive_variable_pairs"
        elif bool(shared_neighbor.any()):
            mechanism_status = "shared_neighbor_bridge_mediated_variable_pairs"
        elif bool(direct_present.any()):
            mechanism_status = "direct_negative_variable_pairs"
        else:
            mechanism_status = "no_graph_mass_variable_pairs"
        meta = object_meta.get(str(object_role_id), {})
        row = {
            "object_role_universe_id": object_role_id,
            "branch": meta.get("branch"),
            "variable_pair_count": int(len(group)),
            "direct_edge_present_pair_count": int(direct_present.sum()),
            "direct_edge_present_pair_share": float(direct_present.mean())
            if len(group)
            else None,
            "direct_positive_pair_count": int(direct_positive.sum()),
            "direct_positive_pair_share": float(direct_positive.mean())
            if len(group)
            else None,
            "direct_positive_gamma3e5_pair_count": int(
                group["direct_positive_at_gamma3e5"].astype(bool).sum()
            ),
            "direct_positive_gamma1e4_pair_count": int(
                group["direct_positive_at_gamma1e4"].astype(bool).sum()
            ),
            "shared_neighbor_pair_count": int(shared_neighbor.sum()),
            "shared_neighbor_pair_share": float(shared_neighbor.mean())
            if len(group)
            else None,
            "mechanism_label_counts": json.dumps(labels, sort_keys=True),
            "mechanism_status": mechanism_status,
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for prefix, column in [
            ("direct_edge_weight", "direct_edge_weight"),
            ("direct_cpm_delta_q", "direct_cpm_delta_q"),
            ("direct_critical_gamma", "direct_critical_gamma"),
            ("common_neighbor_min_weight_sum", "common_neighbor_min_weight_sum"),
            ("common_neighbor_count", "common_neighbor_count"),
            ("left_weighted_degree", "left_weighted_degree"),
            ("right_weighted_degree", "right_weighted_degree"),
        ]:
            row.update(_prefix_stats(prefix, group[column].to_numpy(dtype=np.float64)))
        for scope in ["object_object", "object_support", "support_support"]:
            scope_group = group[group["pair_scope"].astype(str).eq(scope)]
            row[f"{scope}_pair_count"] = int(len(scope_group))
            row[f"{scope}_direct_positive_pair_count"] = int(
                scope_group["direct_positive_at_gamma"].astype(bool).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_summary(
    *,
    difference_dir: Path,
    output_dir: Path,
    pair_frame: pd.DataFrame,
    object_frame: pd.DataFrame,
) -> dict[str, Any]:
    if pair_frame.empty:
        return {
            "schema": "nanoclustering_symmetric_object_variable_pair_graph_mechanism_summary.v1",
            "status": RUN_STATUS,
            "difference_dir": str(difference_dir),
            "output_dir": str(output_dir),
            "object_count": int(len(object_frame)),
            "variable_pair_count": 0,
            "direct_edge_present_pair_count": 0,
            "direct_positive_pair_count": 0,
            "shared_neighbor_pair_count": 0,
            "mechanism_label_counts": {},
            "claim_boundary": CLAIM_BOUNDARY,
        }
    direct_present = pair_frame["direct_edge_present"].astype(bool)
    direct_positive = pair_frame["direct_positive_at_gamma"].astype(bool)
    shared_neighbor = pair_frame["common_neighbor_count"].astype(int) > 0
    summary = {
        "schema": "nanoclustering_symmetric_object_variable_pair_graph_mechanism_summary.v1",
        "status": RUN_STATUS,
        "difference_dir": str(difference_dir),
        "output_dir": str(output_dir),
        "object_count": int(pair_frame["object_role_universe_id"].nunique()),
        "variable_pair_count": int(len(pair_frame)),
        "direct_edge_present_pair_count": int(direct_present.sum()),
        "direct_edge_present_pair_share": float(direct_present.mean()),
        "direct_positive_pair_count": int(direct_positive.sum()),
        "direct_positive_pair_share": float(direct_positive.mean()),
        "direct_positive_gamma3e5_pair_count": int(
            pair_frame["direct_positive_at_gamma3e5"].astype(bool).sum()
        ),
        "direct_positive_gamma1e4_pair_count": int(
            pair_frame["direct_positive_at_gamma1e4"].astype(bool).sum()
        ),
        "shared_neighbor_pair_count": int(shared_neighbor.sum()),
        "shared_neighbor_pair_share": float(shared_neighbor.mean()),
        "mechanism_label_counts": pair_frame["mechanism_label"].value_counts().to_dict(),
        "pair_scope_counts": pair_frame["pair_scope"].value_counts().to_dict(),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for prefix, column in [
        ("direct_edge_weight", "direct_edge_weight"),
        ("direct_cpm_delta_q", "direct_cpm_delta_q"),
        ("direct_critical_gamma", "direct_critical_gamma"),
        ("common_neighbor_min_weight_sum", "common_neighbor_min_weight_sum"),
        ("common_neighbor_count", "common_neighbor_count"),
    ]:
        summary.update(_prefix_stats(prefix, pair_frame[column].to_numpy(dtype=np.float64)))
    return summary


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    object_frame: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Symmetric-Object Variable-Pair Graph Mechanism Review",
        "",
        f"- status: `{summary['status']}`",
        f"- variable_pair_count: {summary['variable_pair_count']}",
        f"- direct_edge_present_pair_count: {summary['direct_edge_present_pair_count']}",
        f"- direct_positive_pair_count: {summary['direct_positive_pair_count']}",
        f"- direct_positive_gamma3e5_pair_count: {summary.get('direct_positive_gamma3e5_pair_count')}",
        f"- shared_neighbor_pair_count: {summary['shared_neighbor_pair_count']}",
        f"- mechanism_label_counts: {summary['mechanism_label_counts']}",
        f"- pair_scope_counts: {summary.get('pair_scope_counts', {})}",
        f"- direct_critical_gamma_max: {summary.get('direct_critical_gamma_max')}",
        f"- common_neighbor_min_weight_sum_median: {summary.get('common_neighbor_min_weight_sum_median')}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Objects",
    ]
    if object_frame.empty:
        lines.append("- no objects")
    else:
        for row in object_frame.sort_values(
            ["direct_positive_pair_count", "shared_neighbor_pair_count", "variable_pair_count"],
            ascending=[False, False, False],
            kind="mergesort",
        ).itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['object_role_universe_id']}: "
                f"variable_pairs={data['variable_pair_count']}, "
                f"direct_present={data['direct_edge_present_pair_count']}, "
                f"direct_positive={data['direct_positive_pair_count']}, "
                f"shared_neighbor_pairs={data['shared_neighbor_pair_count']}, "
                f"status={data['mechanism_status']}, "
                f"direct_gamma_max={data.get('direct_critical_gamma_max')}, "
                f"bridge_min_weight_median={data.get('common_neighbor_min_weight_sum_median')}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "These rows score graph-local evidence for already-observed "
                "variable terminal node-pairs. They are mechanism diagnostics, "
                "not wall/pathway, quality, cost, method-success, or algorithm "
                "claims."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difference-dir", type=Path, default=DEFAULT_DIFFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gamma", type=float, default=1.0e-5)
    parser.add_argument("--edge-chunk-size", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
