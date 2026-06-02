#!/usr/bin/env python3
"""Run a NanoClustering common-mask multistart terminal-multiplicity pilot.

The anchor-release pilots showed that source-anchor and target-anchor terminals
collapse when both are released into the same fixed-outside pair mask. This
runner asks the more primitive question: under that same common feasible set,
do multiple initializations ever converge to different terminal pair states?

It is a terminal-multiplicity diagnostic only. It does not promote wall/pathway
claims, inspect basin quality/cost, claim real-data method success, or claim
algorithm novelty.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_anchor_release_pilot import (
    TARGET_SELECTION_POLICIES,
    _run_leiden_or_hold,
    _select_anchor_release_rows,
)
from run_leiden_basin_nanoclustering_fixed_mask_endpoint_readout_pilot import (
    _route_initial_and_fixed_nodes,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    DEFAULT_BOUNDARY_PLAN_DIR,
    DEFAULT_READINESS_DIR,
    ENDPOINT_TARGET_ROWS_CSV,
    EXECUTION_PLAN_ROWS_CSV,
    GRAPH_INPUT_ROWS_CSV,
    _compact_membership,
    _json_safe,
    _load_graph,
    _load_label_array,
    _mask_for_row,
    _mask_hash,
    _parse_csv_list,
    _read_csv,
    _score_target,
    _source_rows,
    _target_row,
    _union_masks,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_common_mask_multistart_pilot_smoke_20260601"
)

MULTISTART_ATTEMPT_ROWS_CSV = "nanoclustering_common_mask_multistart_attempt_rows.csv"
MULTISTART_PAIR_ROWS_CSV = "nanoclustering_common_mask_multistart_pair_rows.csv"
MULTISTART_CONFIG_JSON = "nanoclustering_common_mask_multistart_config.json"
MULTISTART_SUMMARY_JSON = "nanoclustering_common_mask_multistart_summary.json"
MULTISTART_REPORT_MD = "nanoclustering_common_mask_multistart_report.md"

SOURCE_ARM = "source_anchor_fixed_target_free"
TARGET_ARM = "target_anchor_fixed_source_free"

CLAIM_BOUNDARY = (
    "NanoClustering common-mask multistart terminal-multiplicity pilot only; "
    "runs varied initializations under the same fixed-outside pair mask. It "
    "does not promote wall/pathway claims, inspect basin quality/cost, claim "
    "real-data method success, or claim algorithm novelty."
)
RUN_STATUS = "executed_common_mask_multistart_pilot"


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _two_block_initial(
    *,
    initial_labels: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    initial = np.asarray(initial_labels, dtype=np.uint64).copy()
    next_label = int(initial.max()) + 1
    initial[source_mask] = np.uint64(next_label)
    initial[target_mask] = np.uint64(next_label + 1)
    return initial


def _pair_singleton_initial(
    *,
    initial_labels: np.ndarray,
    pair_mask: np.ndarray,
) -> np.ndarray:
    initial = np.asarray(initial_labels, dtype=np.uint64).copy()
    initial[pair_mask] = np.uint64(int(initial.max()) + 1)
    return initial


def _random_pair_initial(
    *,
    initial_labels: np.ndarray,
    pair_mask: np.ndarray,
    seed: int,
    block_count: int,
) -> np.ndarray:
    initial = np.asarray(initial_labels, dtype=np.uint64).copy()
    pair_count = int(pair_mask.sum())
    if pair_count == 0:
        return initial
    n_blocks = max(1, min(int(block_count), pair_count))
    rng = np.random.default_rng(int(seed))
    labels = rng.integers(0, n_blocks, size=pair_count, dtype=np.uint64)
    initial[pair_mask] = np.uint64(int(initial.max()) + 1) + labels
    return initial


def _common_start_specs(
    *,
    initial_labels: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    pair_mask: np.ndarray,
    method_seed: int,
    random_start_count: int,
    random_block_count: int,
) -> list[dict[str, Any]]:
    target_initial, _, _ = _route_initial_and_fixed_nodes(
        initial_labels=initial_labels,
        source_mask=source_mask,
        target_mask=target_mask,
        route_arm="target_handle_seeded_fixed_outside",
    )
    specs = [
        {
            "start_policy": "source_state",
            "start_index": 0,
            "leiden_seed": int(method_seed),
            "membership": np.asarray(initial_labels, dtype=np.uint64).copy(),
        },
        {
            "start_policy": "target_seeded",
            "start_index": 1,
            "leiden_seed": int(method_seed),
            "membership": target_initial,
        },
        {
            "start_policy": "pair_singleton",
            "start_index": 2,
            "leiden_seed": int(method_seed),
            "membership": _pair_singleton_initial(
                initial_labels=initial_labels,
                pair_mask=pair_mask,
            ),
        },
        {
            "start_policy": "source_target_two_block",
            "start_index": 3,
            "leiden_seed": int(method_seed),
            "membership": _two_block_initial(
                initial_labels=initial_labels,
                source_mask=source_mask,
                target_mask=target_mask,
            ),
        },
    ]
    for idx in range(int(random_start_count)):
        start_seed = int(method_seed) * 1000 + idx + 1
        specs.append(
            {
                "start_policy": f"random_pair_blocks_{idx:03d}",
                "start_index": len(specs),
                "leiden_seed": start_seed,
                "membership": _random_pair_initial(
                    initial_labels=initial_labels,
                    pair_mask=pair_mask,
                    seed=start_seed,
                    block_count=int(random_block_count),
                ),
            }
        )
    return specs


def _pair_summary_rows(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if attempts.empty:
        return pd.DataFrame(rows)
    for pair_id, group in attempts.groupby("multistart_pair_id", sort=False):
        terminal_hashes = sorted(set(group["terminal_pair_hash"].astype(str)))
        rows.append(
            {
                "multistart_pair_id": pair_id,
                "panel_case_id": group["panel_case_id"].iloc[0],
                "panel_case_rank": int(group["panel_case_rank"].iloc[0]),
                "role_id": group["role_id"].iloc[0],
                "role_side": group["role_side"].iloc[0],
                "branch": group["branch"].iloc[0],
                "endpoint_signature_id": group["endpoint_signature_id"].iloc[0],
                "target_handle_id": group["target_handle_id"].iloc[0],
                "method_seed": int(group["method_seed"].iloc[0]),
                "source_mask_hash": group["source_mask_hash"].iloc[0],
                "target_mask_hash": group["target_mask_hash"].iloc[0],
                "pair_mask_hash": group["pair_mask_hash"].iloc[0],
                "source_node_count": int(group["source_node_count"].iloc[0]),
                "target_node_count": int(group["target_node_count"].iloc[0]),
                "pair_node_count": int(group["pair_node_count"].iloc[0]),
                "common_free_node_count": int(group["common_free_node_count"].iloc[0]),
                "start_attempt_count": int(len(group)),
                "executed_attempt_count": int(
                    group["execution_status"].astype(str).str.startswith("executed_").sum()
                ),
                "unique_terminal_pair_hash_count": len(terminal_hashes),
                "terminal_multiplicity_detected": len(terminal_hashes) > 1,
                "terminal_pair_hashes": ";".join(terminal_hashes),
                "target_best_cluster_doc_share_min": float(
                    group["target_best_cluster_doc_share"].min()
                ),
                "target_best_cluster_doc_share_median": float(
                    group["target_best_cluster_doc_share"].median()
                ),
                "target_best_cluster_doc_share_max": float(
                    group["target_best_cluster_doc_share"].max()
                ),
                "source_best_cluster_doc_share_min": float(
                    group["source_best_cluster_doc_share"].min()
                ),
                "source_best_cluster_doc_share_median": float(
                    group["source_best_cluster_doc_share"].median()
                ),
                "source_best_cluster_doc_share_max": float(
                    group["source_best_cluster_doc_share"].max()
                ),
                "quality_min": float(group["quality"].min()),
                "quality_median": float(group["quality"].median()),
                "quality_max": float(group["quality"].max()),
                "seconds_sum": float(group["seconds"].sum()),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    readiness_dir = Path(args.readiness_dir)
    boundary_plan_dir = Path(args.boundary_plan_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint_targets = _read_csv(readiness_dir / ENDPOINT_TARGET_ROWS_CSV)
    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    execution_plan = _read_csv(boundary_plan_dir / EXECUTION_PLAN_ROWS_CSV)
    selected = _select_anchor_release_rows(
        execution_plan,
        case_ranks=_parse_csv_list(args.case_ranks, int),
        method_seeds=_parse_csv_list(args.method_seeds, int),
        max_targets_per_role=int(args.max_targets_per_role),
        max_total_pairs=int(args.max_total_pairs),
        target_selection_policy=str(args.target_selection_policy),
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
    start = time.perf_counter()

    group_keys = ["role_id", "target_handle_id", "method_seed"]
    for _, group in selected.groupby(group_keys, sort=False):
        arm_rows = {str(row["route_arm"]): row for _, row in group.iterrows()}
        if SOURCE_ARM not in arm_rows or TARGET_ARM not in arm_rows:
            continue
        route = arm_rows[SOURCE_ARM]
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
        pair_mask = np.logical_or(source_mask, target_mask)
        common_fixed_nodes = ~pair_mask
        method_seed = int(route["method_seed"])
        pair_id = (
            f"{route['role_id']}__{route['target_handle_id']}__method_seed{method_seed:03d}"
        )
        specs = _common_start_specs(
            initial_labels=initial_labels,
            source_mask=source_mask,
            target_mask=target_mask,
            pair_mask=pair_mask,
            method_seed=method_seed,
            random_start_count=int(args.random_start_count),
            random_block_count=int(args.random_block_count),
        )

        for spec in specs:
            terminal, meta = _run_leiden_or_hold(
                graph=graph,
                membership=spec["membership"],
                fixed_nodes=common_fixed_nodes,
                resolution=float(args.gamma),
                seed=int(spec["leiden_seed"]),
                n_iterations=int(args.n_iterations),
                blocked_status="blocked_empty_common_pair_free_mask",
                executed_status="executed_common_mask_multistart",
            )
            score = _score_target(
                terminal=terminal,
                initial=spec["membership"],
                pair_mask=pair_mask,
                target_mask=target_mask,
                source_mask=source_mask,
                weights=weights,
            )
            row = {
                "multistart_pair_id": pair_id,
                "start_id": f"{pair_id}__{spec['start_policy']}",
                "panel_case_id": route["panel_case_id"],
                "panel_case_rank": int(route["panel_case_rank"]),
                "role_id": route["role_id"],
                "role_side": route["role_side"],
                "branch": branch,
                "endpoint_signature_id": route["endpoint_signature_id"],
                "target_handle_id": route["target_handle_id"],
                "method_seed": method_seed,
                "start_policy": spec["start_policy"],
                "start_index": int(spec["start_index"]),
                "leiden_seed": int(spec["leiden_seed"]),
                "random_block_count": int(args.random_block_count),
                "n_nodes": n_nodes,
                "n_edges": int(graph.n_edges),
                "source_mask_hash": _mask_hash(source_mask),
                "target_mask_hash": _mask_hash(target_mask),
                "pair_mask_hash": _mask_hash(pair_mask),
                "source_node_count": int(source_mask.sum()),
                "target_node_count": int(target_mask.sum()),
                "pair_node_count": int(pair_mask.sum()),
                "common_free_node_count": int(pair_mask.sum()),
                "graph_load_seconds_cached_branch": float(graph_load_seconds),
                **meta,
                "terminal_pair_hash": score["pair_terminal_hash"],
                **score,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
            attempt_rows.append(row)
        attempts = pd.DataFrame(attempt_rows)
        _write_csv(attempts, output_dir / MULTISTART_ATTEMPT_ROWS_CSV)
        _write_csv(_pair_summary_rows(attempts), output_dir / MULTISTART_PAIR_ROWS_CSV)

    attempts = pd.DataFrame(attempt_rows)
    pairs = _pair_summary_rows(attempts)
    _write_csv(attempts, output_dir / MULTISTART_ATTEMPT_ROWS_CSV)
    _write_csv(pairs, output_dir / MULTISTART_PAIR_ROWS_CSV)
    summary = _build_summary(
        selected=selected,
        attempts=attempts,
        pairs=pairs,
        readiness_dir=readiness_dir,
        boundary_plan_dir=boundary_plan_dir,
        output_dir=output_dir,
        elapsed_seconds=time.perf_counter() - start,
    )
    (output_dir / MULTISTART_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_common_mask_multistart_pilot.v1",
        "readiness_dir": str(readiness_dir),
        "boundary_plan_dir": str(boundary_plan_dir),
        "output_dir": str(output_dir),
        "case_ranks": list(_parse_csv_list(args.case_ranks, int)),
        "method_seeds": list(_parse_csv_list(args.method_seeds, int)),
        "max_targets_per_role": int(args.max_targets_per_role),
        "max_total_pairs": int(args.max_total_pairs),
        "target_selection_policy": str(args.target_selection_policy),
        "random_start_count": int(args.random_start_count),
        "random_block_count": int(args.random_block_count),
        "gamma": float(args.gamma),
        "n_iterations": int(args.n_iterations),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / MULTISTART_CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, pairs=pairs)
    return summary


def _build_summary(
    *,
    selected: pd.DataFrame,
    attempts: pd.DataFrame,
    pairs: pd.DataFrame,
    readiness_dir: Path,
    boundary_plan_dir: Path,
    output_dir: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if pairs.empty:
        multiplicity_count = 0
        max_terminal_count = 0
        total_route_seconds = 0.0
    else:
        multiplicity_count = int(pairs["terminal_multiplicity_detected"].sum())
        max_terminal_count = int(pairs["unique_terminal_pair_hash_count"].max())
        total_route_seconds = float(pairs["seconds_sum"].sum())
    return {
        "schema": "nanoclustering_common_mask_multistart_pilot_summary.v1",
        "status": RUN_STATUS if not pairs.empty else "no_multistart_pairs",
        "readiness_dir": str(readiness_dir),
        "boundary_plan_dir": str(boundary_plan_dir),
        "output_dir": str(output_dir),
        "selected_route_attempt_count": int(len(selected)),
        "multistart_pair_count": int(len(pairs)),
        "start_attempt_count": int(len(attempts)),
        "terminal_multiplicity_pair_count": multiplicity_count,
        "terminal_multiplicity_pair_share": (
            float(multiplicity_count / len(pairs)) if len(pairs) else None
        ),
        "max_unique_terminal_pair_hash_count": max_terminal_count,
        "unique_panel_case_count": (
            int(pairs["panel_case_id"].nunique()) if not pairs.empty else 0
        ),
        "unique_target_handle_count": (
            int(pairs["target_handle_id"].nunique()) if not pairs.empty else 0
        ),
        "branch_count": int(pairs["branch"].nunique()) if not pairs.empty else 0,
        "role_count": int(pairs["role_id"].nunique()) if not pairs.empty else 0,
        "common_free_node_count_median": (
            float(pairs["common_free_node_count"].median()) if not pairs.empty else None
        ),
        "common_free_node_count_max": (
            int(pairs["common_free_node_count"].max()) if not pairs.empty else 0
        ),
        "target_best_cluster_doc_share_median": (
            float(pairs["target_best_cluster_doc_share_median"].median())
            if not pairs.empty
            else None
        ),
        "total_route_seconds": total_route_seconds,
        "elapsed_seconds": float(elapsed_seconds),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pairs: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Common-Mask Multistart Pilot",
        "",
        f"- status: `{summary['status']}`",
        f"- multistart_pair_count: {summary['multistart_pair_count']}",
        f"- start_attempt_count: {summary['start_attempt_count']}",
        f"- terminal_multiplicity_pair_count: {summary['terminal_multiplicity_pair_count']}",
        f"- terminal_multiplicity_pair_share: {summary['terminal_multiplicity_pair_share']}",
        f"- max_unique_terminal_pair_hash_count: {summary['max_unique_terminal_pair_hash_count']}",
        f"- common_free_node_count_median: {summary['common_free_node_count_median']}",
        f"- common_free_node_count_max: {summary['common_free_node_count_max']}",
        f"- total_route_seconds: {summary['total_route_seconds']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pairs",
    ]
    if pairs.empty:
        lines.append("- no multistart pairs")
    else:
        for row in pairs.sort_values(
            ["terminal_multiplicity_detected", "panel_case_rank"],
            ascending=[False, True],
        ).itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['multistart_pair_id']}: "
                f"unique_terminals={data['unique_terminal_pair_hash_count']}, "
                f"multiplicity={data['terminal_multiplicity_detected']}, "
                f"free_nodes={data['common_free_node_count']}, "
                f"target_share_median={data['target_best_cluster_doc_share_median']}, "
                f"quality_range=[{data['quality_min']}, {data['quality_max']}]"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "Distinct terminals under the same fixed-outside pair mask are "
                "terminal-multiplicity evidence only. They are not yet wall, "
                "pathway, quality, cost, method-success, or algorithm evidence."
            ),
            "",
        ]
    )
    (output_dir / MULTISTART_REPORT_MD).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--boundary-plan-dir", type=Path, default=DEFAULT_BOUNDARY_PLAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--case-ranks",
        default="4",
        help="Comma-separated panel case ranks. Empty means all selected rows.",
    )
    parser.add_argument("--method-seeds", default="0")
    parser.add_argument("--max-targets-per-role", type=int, default=1)
    parser.add_argument("--max-total-pairs", type=int, default=0)
    parser.add_argument(
        "--target-selection-policy",
        choices=sorted(TARGET_SELECTION_POLICIES),
        default="default",
    )
    parser.add_argument("--random-start-count", type=int, default=4)
    parser.add_argument("--random-block-count", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--n-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
