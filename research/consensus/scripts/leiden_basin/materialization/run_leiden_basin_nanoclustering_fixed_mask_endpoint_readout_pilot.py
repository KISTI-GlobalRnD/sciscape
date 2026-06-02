#!/usr/bin/env python3
"""Run a fixed-mask-aware NanoClustering endpoint readout pilot.

This runner extends the role-local raw route pilot with a bounded constrained
postprocess readout.  For each selected route arm, it:

1. runs the same fixed-outside raw Leiden route;
2. applies a min-nano constrained postprocess approximation where only small
   non-fixed nodes may move;
3. applies a min-doc constrained postprocess approximation under the same
   fixed-node contract;
4. compares whether source/target anchor arms still lead to distinct local
   endpoint readouts.

It is intentionally a readout pilot, not a replacement for the production Rust
postprocess wrapper.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    DEFAULT_BOUNDARY_PLAN_DIR,
    DEFAULT_READINESS_DIR,
    ENDPOINT_TARGET_ROWS_CSV,
    EXECUTION_PLAN_ROWS_CSV,
    GRAPH_INPUT_ROWS_CSV,
    REPO_ROOT,
    _array_hash,
    _compact_membership,
    _json_safe,
    _load_graph,
    _load_label_array,
    _mask_for_row,
    _mask_hash,
    _parse_csv_list,
    _read_csv,
    _score_target,
    _select_plan_rows,
    _source_rows,
    _target_row,
    _union_masks,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_fixed_mask_endpoint_readout_pilot_smoke_20260601"
)

READOUT_ATTEMPT_ROWS_CSV = (
    "nanoclustering_fixed_mask_endpoint_readout_pilot_attempt_rows.csv"
)
READOUT_ROUND_ROWS_CSV = (
    "nanoclustering_fixed_mask_endpoint_readout_pilot_postprocess_round_rows.csv"
)
READOUT_ARM_DISTINCTION_ROWS_CSV = (
    "nanoclustering_fixed_mask_endpoint_readout_pilot_arm_distinction_rows.csv"
)
READOUT_CONFIG_JSON = "nanoclustering_fixed_mask_endpoint_readout_pilot_config.json"
READOUT_SUMMARY_JSON = "nanoclustering_fixed_mask_endpoint_readout_pilot_summary.json"
READOUT_REPORT_MD = "nanoclustering_fixed_mask_endpoint_readout_pilot_report.md"

CLAIM_BOUNDARY = (
    "NanoClustering fixed-mask endpoint readout pilot only; applies a bounded "
    "Python-level constrained postprocess approximation after raw fixed-mask "
    "routes. It does not replace the production postprocess wrapper, promote "
    "wall/pathway claims, judge quality/cost success, or claim a real-data "
    "method result."
)
READOUT_STATUS = "executed_fixed_mask_endpoint_readout_pilot"
WALL_PROMOTION_STATUS = "not_promoted_readout_pilot_only"
QUALITY_COST_STATUS = "excluded_readout_pilot_only"


def _cluster_count_and_weight(
    membership: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(membership, dtype=np.int64)
    counts = np.bincount(labels)
    sums = np.bincount(labels, weights=np.asarray(weights, dtype=np.float64))
    return counts, sums


def _small_node_mask(
    membership: np.ndarray,
    weights: np.ndarray,
    *,
    min_size: int,
    min_weight: float,
) -> tuple[np.ndarray, int, int, float]:
    labels = np.asarray(membership, dtype=np.int64)
    counts, sums = _cluster_count_and_weight(labels, weights)
    if min_weight > 0.0:
        small_clusters = sums < float(min_weight)
    elif min_size > 0:
        small_clusters = counts < int(min_size)
    else:
        small_clusters = np.zeros_like(counts, dtype=np.bool_)
    small_nodes = small_clusters[labels]
    return (
        small_nodes,
        int(small_clusters.sum()),
        int(small_nodes.sum()),
        float(sums[small_clusters].sum()) if len(sums) else 0.0,
    )


def _constrained_postprocess_readout(
    *,
    graph: Any,
    membership: np.ndarray,
    base_fixed_nodes: np.ndarray,
    weights: np.ndarray,
    resolution: float,
    seed: int,
    n_iterations: int,
    max_rounds: int,
    min_size: int,
    min_weight: float,
    stage: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    current = _compact_membership(membership)
    fixed_base = np.asarray(base_fixed_nodes, dtype=np.bool_)
    round_rows: list[dict[str, Any]] = []

    for round_index in range(int(max_rounds)):
        small_nodes, small_cluster_count, small_node_count, small_weight = _small_node_mask(
            current,
            weights,
            min_size=int(min_size),
            min_weight=float(min_weight),
        )
        mutable_nodes = np.logical_and(small_nodes, ~fixed_base)
        mutable_count = int(mutable_nodes.sum())
        before_hash = _array_hash(current)
        if mutable_count == 0:
            round_rows.append(
                {
                    "postprocess_stage": stage,
                    "round_index": round_index,
                    "round_execution_status": "skipped_no_mutable_small_nodes",
                    "min_size": int(min_size),
                    "min_weight": float(min_weight),
                    "small_cluster_count": small_cluster_count,
                    "small_node_count": small_node_count,
                    "small_doc_sum": small_weight,
                    "mutable_node_count": 0,
                    "changed_node_count": 0,
                    "before_hash": before_hash,
                    "after_hash": before_hash,
                    "round_seconds": 0.0,
                    "quality": None,
                    "n_clusters": int(np.unique(current).size),
                }
            )
            break

        fixed_nodes = np.logical_or(fixed_base, ~mutable_nodes)
        start = time.perf_counter()
        result = graph.run_leiden(
            resolution=float(resolution),
            n_iterations=int(n_iterations),
            seed=int(seed),
            initial_membership=np.asarray(current, dtype=np.uint64),
            fixed_nodes=fixed_nodes,
        )
        seconds = time.perf_counter() - start
        next_membership = _compact_membership(result.membership)
        after_hash = _array_hash(next_membership)
        changed = int(np.count_nonzero(next_membership != current))
        round_rows.append(
            {
                "postprocess_stage": stage,
                "round_index": round_index,
                "round_execution_status": "executed_constrained_small_cluster_leiden",
                "min_size": int(min_size),
                "min_weight": float(min_weight),
                "small_cluster_count": small_cluster_count,
                "small_node_count": small_node_count,
                "small_doc_sum": small_weight,
                "mutable_node_count": mutable_count,
                "changed_node_count": changed,
                "before_hash": before_hash,
                "after_hash": after_hash,
                "round_seconds": float(seconds),
                "quality": float(result.quality),
                "n_clusters": int(result.n_clusters),
            }
        )
        current = next_membership
        if changed == 0 or before_hash == after_hash:
            break

    return current, round_rows


def _prefix_score(prefix: str, score: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in score.items()}


def _route_initial_and_fixed_nodes(
    *,
    initial_labels: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    route_arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_mask = np.logical_or(source_mask, target_mask)
    fixed_nodes = ~pair_mask
    initial = np.asarray(initial_labels, dtype=np.uint64).copy()
    if route_arm == "target_handle_seeded_fixed_outside":
        initial[target_mask] = np.uint64(initial.max() + 1)
    elif route_arm == "target_anchor_fixed_source_free":
        initial[target_mask] = np.uint64(initial.max() + 1)
        fixed_nodes = np.logical_or(~pair_mask, target_mask)
    elif route_arm == "source_anchor_fixed_target_free":
        fixed_nodes = np.logical_or(~pair_mask, source_mask)
    elif route_arm != "source_state_fixed_outside_control":
        raise ValueError(f"unsupported route arm: {route_arm}")
    return initial, fixed_nodes, pair_mask


def run(args: argparse.Namespace) -> dict[str, Any]:
    readiness_dir = Path(args.readiness_dir)
    boundary_plan_dir = Path(args.boundary_plan_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint_targets = _read_csv(readiness_dir / ENDPOINT_TARGET_ROWS_CSV)
    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    execution_plan = _read_csv(boundary_plan_dir / EXECUTION_PLAN_ROWS_CSV)
    selected = _select_plan_rows(
        execution_plan,
        case_ranks=_parse_csv_list(args.case_ranks, int),
        method_seeds=_parse_csv_list(args.method_seeds, int),
        route_arms=_parse_csv_list(args.route_arms, str),
        max_targets_per_role=int(args.max_targets_per_role),
    )

    graph_by_branch = {
        str(row["branch"]): row
        for _, row in graph_rows.iterrows()
        if str(row.get("runtime_graph_status", "")).startswith("ready_")
    }
    manifest_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, float]] = {}

    attempt_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []

    for _, route in selected.iterrows():
        branch = str(route["branch"])
        if branch not in graph_cache:
            graph, weights, load_seconds = _load_graph(
                graph_by_branch[branch],
                manifest_cache,
            )
            graph_cache[branch] = graph, weights, load_seconds
        graph, weights, graph_load_seconds = graph_cache[branch]
        n_nodes = int(graph.n_nodes)

        source = _source_rows(endpoint_targets, str(route["endpoint_signature_id"]))
        if source.empty:
            raise ValueError(f"missing source handle for {route['endpoint_signature_id']}")
        source_full_path = Path(str(source.iloc[0]["membership_path"]))
        label_col = str(source.iloc[0]["label_cols"]).split(";")[0] or "candidate_micro_id"
        initial_labels = _compact_membership(_load_label_array(source_full_path, label_col))
        source_mask = _union_masks(source, label_cache, n_nodes)
        target = _target_row(endpoint_targets, str(route["target_handle_id"]))
        target_mask = _mask_for_row(target, label_cache)
        route_arm = str(route["route_arm"])
        initial, fixed_nodes, pair_mask = _route_initial_and_fixed_nodes(
            initial_labels=initial_labels,
            source_mask=source_mask,
            target_mask=target_mask,
            route_arm=route_arm,
        )

        free_count = int((~fixed_nodes).sum())
        if free_count > 0:
            start = time.perf_counter()
            raw = graph.run_leiden(
                resolution=float(args.gamma),
                n_iterations=int(args.n_iterations),
                seed=int(route["method_seed"]),
                initial_membership=initial,
                fixed_nodes=fixed_nodes,
            )
            raw_seconds = time.perf_counter() - start
            raw_terminal = _compact_membership(raw.membership)
            raw_status = "executed_role_local_fixed_mask_raw_route"
            raw_quality = float(raw.quality)
            raw_n_clusters = int(raw.n_clusters)
        else:
            raw_seconds = 0.0
            raw_terminal = _compact_membership(initial)
            raw_status = "blocked_empty_free_mask_before_rust"
            raw_quality = None
            raw_n_clusters = int(np.unique(raw_terminal).size)

        post_nano, nano_rounds = _constrained_postprocess_readout(
            graph=graph,
            membership=raw_terminal,
            base_fixed_nodes=fixed_nodes,
            weights=weights,
            resolution=float(args.gamma),
            seed=int(route["method_seed"]),
            n_iterations=int(args.post_iterations),
            max_rounds=int(args.post_max_rounds),
            min_size=int(args.min_nano),
            min_weight=0.0,
            stage="min_nano",
        )
        post_doc, doc_rounds = _constrained_postprocess_readout(
            graph=graph,
            membership=post_nano,
            base_fixed_nodes=fixed_nodes,
            weights=weights,
            resolution=float(args.gamma),
            seed=int(route["method_seed"]),
            n_iterations=int(args.post_iterations),
            max_rounds=int(args.post_max_rounds),
            min_size=0,
            min_weight=float(args.min_docs),
            stage="min_docs",
        )

        route_attempt_id = str(route["route_attempt_id"])
        for row in nano_rounds + doc_rounds:
            round_rows.append(
                {
                    "route_attempt_id": route_attempt_id,
                    "panel_case_id": route["panel_case_id"],
                    "role_id": route["role_id"],
                    "role_side": route["role_side"],
                    "branch": branch,
                    "target_handle_id": route["target_handle_id"],
                    "method_seed": int(route["method_seed"]),
                    "route_arm": route_arm,
                    **row,
                }
            )

        raw_score = _score_target(
            terminal=raw_terminal,
            initial=initial,
            pair_mask=pair_mask,
            target_mask=target_mask,
            source_mask=source_mask,
            weights=weights,
        )
        nano_score = _score_target(
            terminal=post_nano,
            initial=raw_terminal,
            pair_mask=pair_mask,
            target_mask=target_mask,
            source_mask=source_mask,
            weights=weights,
        )
        doc_score = _score_target(
            terminal=post_doc,
            initial=raw_terminal,
            pair_mask=pair_mask,
            target_mask=target_mask,
            source_mask=source_mask,
            weights=weights,
        )

        nano_changed = int(sum(row["changed_node_count"] for row in nano_rounds))
        doc_changed = int(sum(row["changed_node_count"] for row in doc_rounds))
        nano_seconds = float(sum(row["round_seconds"] for row in nano_rounds))
        doc_seconds = float(sum(row["round_seconds"] for row in doc_rounds))

        attempt_rows.append(
            {
                "route_attempt_id": route_attempt_id,
                "panel_case_id": route["panel_case_id"],
                "panel_case_rank": int(route["panel_case_rank"]),
                "role_id": route["role_id"],
                "role_side": route["role_side"],
                "branch": branch,
                "endpoint_signature_id": route["endpoint_signature_id"],
                "target_handle_id": route["target_handle_id"],
                "method_seed": int(route["method_seed"]),
                "route_arm": route_arm,
                "n_nodes": n_nodes,
                "n_edges": int(graph.n_edges),
                "source_mask_hash": _mask_hash(source_mask),
                "target_mask_hash": _mask_hash(target_mask),
                "pair_mask_hash": _mask_hash(pair_mask),
                "pair_node_count": int(pair_mask.sum()),
                "fixed_node_count": int(fixed_nodes.sum()),
                "free_node_count": free_count,
                "raw_route_status": raw_status,
                "raw_n_clusters": raw_n_clusters,
                "raw_quality": raw_quality,
                "raw_seconds": float(raw_seconds),
                "post_nano_round_count": len(nano_rounds),
                "post_doc_round_count": len(doc_rounds),
                "post_nano_changed_node_count": nano_changed,
                "post_doc_changed_node_count": doc_changed,
                "post_nano_seconds": nano_seconds,
                "post_doc_seconds": doc_seconds,
                "graph_load_seconds_cached_branch": float(graph_load_seconds),
                **_prefix_score("raw", raw_score),
                **_prefix_score("post_nano", nano_score),
                **_prefix_score("post_doc", doc_score),
                "readout_execution_status": READOUT_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        _write_csv(pd.DataFrame(attempt_rows), output_dir / READOUT_ATTEMPT_ROWS_CSV)
        _write_csv(pd.DataFrame(round_rows), output_dir / READOUT_ROUND_ROWS_CSV)

    attempts = pd.DataFrame(attempt_rows)
    rounds = pd.DataFrame(round_rows)
    arm_distinctions = _build_arm_distinctions(attempts)
    _write_csv(arm_distinctions, output_dir / READOUT_ARM_DISTINCTION_ROWS_CSV)
    _write_csv(attempts, output_dir / READOUT_ATTEMPT_ROWS_CSV)
    _write_csv(rounds, output_dir / READOUT_ROUND_ROWS_CSV)

    summary = _build_summary(
        selected=selected,
        attempts=attempts,
        rounds=rounds,
        arm_distinctions=arm_distinctions,
        readiness_dir=readiness_dir,
        boundary_plan_dir=boundary_plan_dir,
        output_dir=output_dir,
    )
    (output_dir / READOUT_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_fixed_mask_endpoint_readout_pilot.v1",
        "readiness_dir": str(readiness_dir),
        "boundary_plan_dir": str(boundary_plan_dir),
        "output_dir": str(output_dir),
        "case_ranks": list(_parse_csv_list(args.case_ranks, int)),
        "method_seeds": list(_parse_csv_list(args.method_seeds, int)),
        "route_arms": list(_parse_csv_list(args.route_arms, str)),
        "max_targets_per_role": int(args.max_targets_per_role),
        "gamma": float(args.gamma),
        "min_nano": int(args.min_nano),
        "min_docs": float(args.min_docs),
        "n_iterations": int(args.n_iterations),
        "post_iterations": int(args.post_iterations),
        "post_max_rounds": int(args.post_max_rounds),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / READOUT_CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, attempts=attempts)
    return summary


def _build_arm_distinctions(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["role_id", "target_handle_id", "method_seed"]
    for key, group in attempts.groupby(keys, sort=False):
        row: dict[str, Any] = {
            "role_id": key[0],
            "target_handle_id": key[1],
            "method_seed": int(key[2]),
            "route_arm_count": int(group["route_arm"].nunique()),
        }
        for stage, hash_col in [
            ("raw", "raw_pair_terminal_hash"),
            ("post_nano", "post_nano_pair_terminal_hash"),
            ("post_doc", "post_doc_pair_terminal_hash"),
        ]:
            hashes = group.set_index("route_arm")[hash_col].to_dict()
            unique_count = int(group[hash_col].nunique(dropna=True))
            row[f"{stage}_terminal_hash_unique_count"] = unique_count
            row[f"{stage}_route_arm_hashes"] = ";".join(
                f"{arm}={hash_value}" for arm, hash_value in sorted(hashes.items())
            )
            row[f"{stage}_route_arms_distinct"] = unique_count > 1
        rows.append(row)
    return pd.DataFrame(rows)


def _build_summary(
    *,
    selected: pd.DataFrame,
    attempts: pd.DataFrame,
    rounds: pd.DataFrame,
    arm_distinctions: pd.DataFrame,
    readiness_dir: Path,
    boundary_plan_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if attempts.empty:
        raw_distinct = nano_distinct = doc_distinct = 0
        post_doc_changed_count = 0
        raw_to_post_nano_changed = 0
        raw_to_post_doc_changed = 0
        blocked_empty = 0
    else:
        raw_distinct = int(arm_distinctions["raw_route_arms_distinct"].sum())
        nano_distinct = int(arm_distinctions["post_nano_route_arms_distinct"].sum())
        doc_distinct = int(arm_distinctions["post_doc_route_arms_distinct"].sum())
        post_doc_changed_count = int(
            attempts["post_doc_changed_node_count"].astype(int).gt(0).sum()
        )
        raw_to_post_nano_changed = int(
            attempts["raw_pair_terminal_hash"]
            .astype(str)
            .ne(attempts["post_nano_pair_terminal_hash"].astype(str))
            .sum()
        )
        raw_to_post_doc_changed = int(
            attempts["raw_pair_terminal_hash"]
            .astype(str)
            .ne(attempts["post_doc_pair_terminal_hash"].astype(str))
            .sum()
        )
        blocked_empty = int(
            attempts["raw_route_status"].astype(str).eq(
                "blocked_empty_free_mask_before_rust"
            ).sum()
        )
    executed_rounds = 0
    if not rounds.empty:
        executed_rounds = int(
            rounds["round_execution_status"]
            .astype(str)
            .eq("executed_constrained_small_cluster_leiden")
            .sum()
        )
    return {
        "schema": "nanoclustering_fixed_mask_endpoint_readout_pilot_summary.v1",
        "status": READOUT_STATUS if not attempts.empty else "no_selected_attempts",
        "readiness_dir": str(readiness_dir),
        "boundary_plan_dir": str(boundary_plan_dir),
        "output_dir": str(output_dir),
        "selected_route_attempt_count": int(len(selected)),
        "executed_route_attempt_count": int(len(attempts)),
        "postprocess_round_row_count": int(len(rounds)),
        "executed_postprocess_round_count": executed_rounds,
        "branch_count": int(attempts["branch"].nunique()) if not attempts.empty else 0,
        "role_count": int(attempts["role_id"].nunique()) if not attempts.empty else 0,
        "target_handle_count": (
            int(attempts["target_handle_id"].nunique()) if not attempts.empty else 0
        ),
        "arm_distinction_pair_count": int(len(arm_distinctions)),
        "raw_route_arms_distinct_count": raw_distinct,
        "post_nano_route_arms_distinct_count": nano_distinct,
        "post_doc_route_arms_distinct_count": doc_distinct,
        "post_doc_changed_attempt_count": post_doc_changed_count,
        "raw_to_post_nano_pair_hash_changed_count": raw_to_post_nano_changed,
        "raw_to_post_doc_pair_hash_changed_count": raw_to_post_doc_changed,
        "blocked_empty_free_mask_count": blocked_empty,
        "total_raw_seconds": (
            float(attempts["raw_seconds"].sum()) if not attempts.empty else 0.0
        ),
        "total_post_nano_seconds": (
            float(attempts["post_nano_seconds"].sum()) if not attempts.empty else 0.0
        ),
        "total_post_doc_seconds": (
            float(attempts["post_doc_seconds"].sum()) if not attempts.empty else 0.0
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    attempts: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Fixed-Mask Endpoint Readout Pilot",
        "",
        f"- status: `{summary['status']}`",
        f"- selected_route_attempt_count: {summary['selected_route_attempt_count']}",
        f"- executed_route_attempt_count: {summary['executed_route_attempt_count']}",
        f"- raw_route_arms_distinct_count: {summary['raw_route_arms_distinct_count']}",
        f"- post_nano_route_arms_distinct_count: {summary['post_nano_route_arms_distinct_count']}",
        f"- post_doc_route_arms_distinct_count: {summary['post_doc_route_arms_distinct_count']}",
        f"- executed_postprocess_round_count: {summary['executed_postprocess_round_count']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Attempts",
    ]
    if attempts.empty:
        lines.append("- no attempts executed")
    else:
        for row in attempts.sort_values(["role_side", "route_arm"]).itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['route_attempt_id']}: "
                f"free={data['free_node_count']}, "
                f"raw_hash={data['raw_pair_terminal_hash']}, "
                f"post_doc_hash={data['post_doc_pair_terminal_hash']}, "
                f"post_doc_target_share={data['post_doc_target_best_cluster_doc_share']}, "
                f"post_doc_source_share={data['post_doc_source_best_cluster_doc_share']}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This readout keeps the route fixed-mask contract through a bounded "
                "postprocess approximation. It is not the production NanoClustering "
                "endpoint wrapper and does not promote wall/pathway or method-success "
                "claims."
            ),
            "",
        ]
    )
    (output_dir / READOUT_REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--boundary-plan-dir", type=Path, default=DEFAULT_BOUNDARY_PLAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ranks", default="4")
    parser.add_argument("--method-seeds", default="0")
    parser.add_argument(
        "--route-arms",
        default="source_anchor_fixed_target_free,target_anchor_fixed_source_free",
    )
    parser.add_argument("--max-targets-per-role", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--min-nano", type=int, default=3)
    parser.add_argument("--min-docs", type=float, default=3000.0)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--post-iterations", type=int, default=2)
    parser.add_argument("--post-max-rounds", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
