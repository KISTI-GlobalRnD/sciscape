#!/usr/bin/env python3
"""Run a NanoClustering anchor-release pilot.

This is the next wall-facing gate after the fixed-mask endpoint readout.  The
previous readout showed source-anchor and target-anchor local states are stable
under a constrained postprocess approximation, but that test reused a subset of
the raw-route free set.  This runner performs a less tautological check:

1. produce the two anchored local terminals for the same source/target pair;
2. release both terminals into the same fixed-outside pair mask;
3. test whether they collapse to one terminal or remain distinct under the
   common feasible set.

It is a route diagnostic only.  It does not promote wall/pathway claims,
quality/cost claims, real-data method success, or algorithm novelty.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    / "leiden_basin_nanoclustering_anchor_release_pilot_smoke_20260601"
)

ANCHOR_RELEASE_PAIR_ROWS_CSV = "nanoclustering_anchor_release_pilot_pair_rows.csv"
ANCHOR_RELEASE_CONFIG_JSON = "nanoclustering_anchor_release_pilot_config.json"
ANCHOR_RELEASE_SUMMARY_JSON = "nanoclustering_anchor_release_pilot_summary.json"
ANCHOR_RELEASE_REPORT_MD = "nanoclustering_anchor_release_pilot_report.md"

SOURCE_ARM = "source_anchor_fixed_target_free"
TARGET_ARM = "target_anchor_fixed_source_free"

CLAIM_BOUNDARY = (
    "NanoClustering anchor-release pilot only; compares source-anchor and "
    "target-anchor local terminals after releasing both into the same "
    "fixed-outside pair mask. It does not promote wall/pathway claims, "
    "quality/cost claims, real-data method success, or algorithm novelty."
)
RUN_STATUS = "executed_anchor_release_pilot"
TARGET_SELECTION_POLICIES = {
    "default",
    "largest_pair_free",
    "largest_pair_doc",
    "lowest_target_overlap",
    "largest_target_doc",
}


def _run_leiden_or_hold(
    *,
    graph: Any,
    membership: np.ndarray,
    fixed_nodes: np.ndarray,
    resolution: float,
    seed: int,
    n_iterations: int,
    blocked_status: str,
    executed_status: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    free_count = int((~np.asarray(fixed_nodes, dtype=np.bool_)).sum())
    if free_count == 0:
        terminal = _compact_membership(membership)
        return terminal, {
            "execution_status": blocked_status,
            "free_node_count": 0,
            "seconds": 0.0,
            "quality": None,
            "n_clusters": int(np.unique(terminal).size),
        }
    start = time.perf_counter()
    result = graph.run_leiden(
        resolution=float(resolution),
        n_iterations=int(n_iterations),
        seed=int(seed),
        initial_membership=np.asarray(membership, dtype=np.uint64),
        fixed_nodes=np.asarray(fixed_nodes, dtype=np.bool_),
    )
    seconds = time.perf_counter() - start
    terminal = _compact_membership(result.membership)
    return terminal, {
        "execution_status": executed_status,
        "free_node_count": free_count,
        "seconds": float(seconds),
        "quality": float(result.quality),
        "n_clusters": int(result.n_clusters),
    }


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _select_anchor_release_rows(
    execution_plan: pd.DataFrame,
    *,
    case_ranks: tuple[int, ...],
    method_seeds: tuple[int, ...],
    max_targets_per_role: int,
    max_total_pairs: int,
    target_selection_policy: str,
) -> pd.DataFrame:
    if target_selection_policy not in TARGET_SELECTION_POLICIES:
        raise ValueError(
            "unsupported target selection policy: "
            f"{target_selection_policy}; expected one of "
            f"{sorted(TARGET_SELECTION_POLICIES)}"
        )
    selected = _select_plan_rows(
        execution_plan,
        case_ranks=case_ranks,
        method_seeds=method_seeds,
        route_arms=(SOURCE_ARM, TARGET_ARM),
        max_targets_per_role=0,
    )
    pair_keys = ["role_id", "target_handle_id", "method_seed"]
    pair_rows = (
        selected[pair_keys + [
            "panel_case_rank",
            "role_side",
            "branch",
            "target_seed",
            "pair_free_node_count",
            "pair_free_doc_sum",
            "target_doc_sum",
            "source_target_overlap_target_doc_share",
        ]]
        .drop_duplicates(pair_keys)
        .copy()
    )
    if target_selection_policy == "largest_pair_free":
        sort_cols = ["pair_free_node_count", "pair_free_doc_sum", "panel_case_rank"]
        ascending = [False, False, True]
    elif target_selection_policy == "largest_pair_doc":
        sort_cols = ["pair_free_doc_sum", "pair_free_node_count", "panel_case_rank"]
        ascending = [False, False, True]
    elif target_selection_policy == "lowest_target_overlap":
        sort_cols = [
            "source_target_overlap_target_doc_share",
            "pair_free_node_count",
            "panel_case_rank",
        ]
        ascending = [True, False, True]
    elif target_selection_policy == "largest_target_doc":
        sort_cols = ["target_doc_sum", "pair_free_node_count", "panel_case_rank"]
        ascending = [False, False, True]
    else:
        sort_cols = ["panel_case_rank", "role_side", "target_seed"]
        ascending = [True, True, True]
    pair_rows = pair_rows.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    if max_targets_per_role > 0:
        pair_rows = pair_rows.groupby("role_id", sort=False).head(max_targets_per_role)
    if max_total_pairs > 0:
        pair_rows = pair_rows.head(max_total_pairs)
    keep = pair_rows[pair_keys]
    return selected.merge(keep, on=pair_keys, how="inner").reset_index(drop=True)


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
    pair_rows: list[dict[str, Any]] = []

    group_keys = ["role_id", "target_handle_id", "method_seed"]
    for key, group in selected.groupby(group_keys, sort=False):
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

        source_initial, source_fixed, source_pair_mask = _route_initial_and_fixed_nodes(
            initial_labels=initial_labels,
            source_mask=source_mask,
            target_mask=target_mask,
            route_arm=SOURCE_ARM,
        )
        target_initial, target_fixed, target_pair_mask = _route_initial_and_fixed_nodes(
            initial_labels=initial_labels,
            source_mask=source_mask,
            target_mask=target_mask,
            route_arm=TARGET_ARM,
        )
        if not np.array_equal(pair_mask, source_pair_mask) or not np.array_equal(
            pair_mask,
            target_pair_mask,
        ):
            raise ValueError("route arms disagree on pair mask")

        method_seed = int(route["method_seed"])
        source_anchor_terminal, source_anchor_meta = _run_leiden_or_hold(
            graph=graph,
            membership=source_initial,
            fixed_nodes=source_fixed,
            resolution=float(args.gamma),
            seed=method_seed,
            n_iterations=int(args.n_iterations),
            blocked_status="blocked_empty_source_anchor_free_mask",
            executed_status="executed_source_anchor_route",
        )
        target_anchor_terminal, target_anchor_meta = _run_leiden_or_hold(
            graph=graph,
            membership=target_initial,
            fixed_nodes=target_fixed,
            resolution=float(args.gamma),
            seed=method_seed,
            n_iterations=int(args.n_iterations),
            blocked_status="blocked_empty_target_anchor_free_mask",
            executed_status="executed_target_anchor_route",
        )

        source_release_terminal, source_release_meta = _run_leiden_or_hold(
            graph=graph,
            membership=source_anchor_terminal,
            fixed_nodes=common_fixed_nodes,
            resolution=float(args.gamma),
            seed=method_seed,
            n_iterations=int(args.release_iterations),
            blocked_status="blocked_empty_common_pair_free_mask",
            executed_status="executed_source_terminal_common_release",
        )
        target_release_terminal, target_release_meta = _run_leiden_or_hold(
            graph=graph,
            membership=target_anchor_terminal,
            fixed_nodes=common_fixed_nodes,
            resolution=float(args.gamma),
            seed=method_seed,
            n_iterations=int(args.release_iterations),
            blocked_status="blocked_empty_common_pair_free_mask",
            executed_status="executed_target_terminal_common_release",
        )

        source_anchor_score = _score_target(
            terminal=source_anchor_terminal,
            initial=source_initial,
            pair_mask=pair_mask,
            target_mask=target_mask,
            source_mask=source_mask,
            weights=weights,
        )
        target_anchor_score = _score_target(
            terminal=target_anchor_terminal,
            initial=target_initial,
            pair_mask=pair_mask,
            target_mask=target_mask,
            source_mask=source_mask,
            weights=weights,
        )
        source_release_score = _score_target(
            terminal=source_release_terminal,
            initial=source_anchor_terminal,
            pair_mask=pair_mask,
            target_mask=target_mask,
            source_mask=source_mask,
            weights=weights,
        )
        target_release_score = _score_target(
            terminal=target_release_terminal,
            initial=target_anchor_terminal,
            pair_mask=pair_mask,
            target_mask=target_mask,
            source_mask=source_mask,
            weights=weights,
        )

        source_anchor_hash = source_anchor_score["pair_terminal_hash"]
        target_anchor_hash = target_anchor_score["pair_terminal_hash"]
        source_release_hash = source_release_score["pair_terminal_hash"]
        target_release_hash = target_release_score["pair_terminal_hash"]
        anchor_distinct = source_anchor_hash != target_anchor_hash
        release_distinct = source_release_hash != target_release_hash
        if not anchor_distinct:
            release_status = "not_evaluated_anchor_terminals_already_collapsed"
        elif release_distinct:
            release_status = "released_terminals_remain_distinct"
        else:
            release_status = "released_terminals_collapsed"

        row = {
            "release_pair_id": (
                f"{route['role_id']}__{route['target_handle_id']}__method_seed{method_seed:03d}"
            ),
            "panel_case_id": route["panel_case_id"],
            "panel_case_rank": int(route["panel_case_rank"]),
            "role_id": route["role_id"],
            "role_side": route["role_side"],
            "branch": branch,
            "endpoint_signature_id": route["endpoint_signature_id"],
            "target_handle_id": route["target_handle_id"],
            "method_seed": method_seed,
            "n_nodes": n_nodes,
            "n_edges": int(graph.n_edges),
            "source_mask_hash": _mask_hash(source_mask),
            "target_mask_hash": _mask_hash(target_mask),
            "pair_mask_hash": _mask_hash(pair_mask),
            "source_node_count": int(source_mask.sum()),
            "target_node_count": int(target_mask.sum()),
            "pair_node_count": int(pair_mask.sum()),
            "common_release_free_node_count": int(pair_mask.sum()),
            "graph_load_seconds_cached_branch": float(graph_load_seconds),
            **_prefix("source_anchor", source_anchor_meta),
            **_prefix("target_anchor", target_anchor_meta),
            **_prefix("source_release", source_release_meta),
            **_prefix("target_release", target_release_meta),
            "source_anchor_pair_hash": source_anchor_hash,
            "target_anchor_pair_hash": target_anchor_hash,
            "source_release_pair_hash": source_release_hash,
            "target_release_pair_hash": target_release_hash,
            "anchor_pair_distinct": anchor_distinct,
            "release_pair_distinct": release_distinct,
            "release_collapse_status": release_status,
            **_prefix("source_anchor", source_anchor_score),
            **_prefix("target_anchor", target_anchor_score),
            **_prefix("source_release", source_release_score),
            **_prefix("target_release", target_release_score),
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        pair_rows.append(row)
        _write_csv(pd.DataFrame(pair_rows), output_dir / ANCHOR_RELEASE_PAIR_ROWS_CSV)

    pairs = pd.DataFrame(pair_rows)
    _write_csv(pairs, output_dir / ANCHOR_RELEASE_PAIR_ROWS_CSV)
    summary = _build_summary(
        selected=selected,
        pairs=pairs,
        readiness_dir=readiness_dir,
        boundary_plan_dir=boundary_plan_dir,
        output_dir=output_dir,
    )
    (output_dir / ANCHOR_RELEASE_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_anchor_release_pilot.v1",
        "readiness_dir": str(readiness_dir),
        "boundary_plan_dir": str(boundary_plan_dir),
        "output_dir": str(output_dir),
        "case_ranks": list(_parse_csv_list(args.case_ranks, int)),
        "method_seeds": list(_parse_csv_list(args.method_seeds, int)),
        "max_targets_per_role": int(args.max_targets_per_role),
        "max_total_pairs": int(args.max_total_pairs),
        "target_selection_policy": str(args.target_selection_policy),
        "gamma": float(args.gamma),
        "n_iterations": int(args.n_iterations),
        "release_iterations": int(args.release_iterations),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / ANCHOR_RELEASE_CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, pairs=pairs)
    return summary


def _build_summary(
    *,
    selected: pd.DataFrame,
    pairs: pd.DataFrame,
    readiness_dir: Path,
    boundary_plan_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if pairs.empty:
        anchor_distinct = 0
        release_distinct = 0
        release_collapsed = 0
        unique_local_objects = 0
        total_seconds = 0.0
    else:
        anchor_distinct = int(pairs["anchor_pair_distinct"].sum())
        release_distinct = int(pairs["release_pair_distinct"].sum())
        release_collapsed = int(
            pairs["release_collapse_status"]
            .astype(str)
            .eq("released_terminals_collapsed")
            .sum()
        )
        unique_local_objects = int(
            pairs[
                ["source_mask_hash", "target_mask_hash", "pair_mask_hash"]
            ].drop_duplicates().shape[0]
        )
        total_seconds = float(
            pairs[
                [
                    "source_anchor_seconds",
                    "target_anchor_seconds",
                    "source_release_seconds",
                    "target_release_seconds",
                ]
            ].sum().sum()
        )
    return {
        "schema": "nanoclustering_anchor_release_pilot_summary.v1",
        "status": RUN_STATUS if not pairs.empty else "no_release_pairs",
        "readiness_dir": str(readiness_dir),
        "boundary_plan_dir": str(boundary_plan_dir),
        "output_dir": str(output_dir),
        "selected_route_attempt_count": int(len(selected)),
        "release_pair_count": int(len(pairs)),
        "unique_panel_case_count": (
            int(pairs["panel_case_id"].nunique()) if not pairs.empty else 0
        ),
        "unique_target_handle_count": (
            int(pairs["target_handle_id"].nunique()) if not pairs.empty else 0
        ),
        "unique_local_source_target_pair_mask_object_count": unique_local_objects,
        "anchor_pair_distinct_count": anchor_distinct,
        "release_pair_distinct_count": release_distinct,
        "release_pair_collapsed_count": release_collapsed,
        "release_pair_distinct_share": (
            float(release_distinct / len(pairs)) if len(pairs) else None
        ),
        "branch_count": int(pairs["branch"].nunique()) if not pairs.empty else 0,
        "role_count": int(pairs["role_id"].nunique()) if not pairs.empty else 0,
        "total_route_seconds": total_seconds,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pairs: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Anchor-Release Pilot",
        "",
        f"- status: `{summary['status']}`",
        f"- release_pair_count: {summary['release_pair_count']}",
        f"- unique_local_source_target_pair_mask_object_count: {summary['unique_local_source_target_pair_mask_object_count']}",
        f"- anchor_pair_distinct_count: {summary['anchor_pair_distinct_count']}",
        f"- release_pair_distinct_count: {summary['release_pair_distinct_count']}",
        f"- release_pair_collapsed_count: {summary['release_pair_collapsed_count']}",
        f"- release_pair_distinct_share: {summary['release_pair_distinct_share']}",
        f"- total_route_seconds: {summary['total_route_seconds']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pairs",
    ]
    if pairs.empty:
        lines.append("- no release pairs")
    else:
        for row in pairs.sort_values(["panel_case_rank", "role_side"]).itertuples(
            index=False
        ):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['release_pair_id']}: "
                f"anchor_distinct={data['anchor_pair_distinct']}, "
                f"release_distinct={data['release_pair_distinct']}, "
                f"status={data['release_collapse_status']}, "
                f"target_share(source_release)="
                f"{data['source_release_target_best_cluster_doc_share']}, "
                f"target_share(target_release)="
                f"{data['target_release_target_best_cluster_doc_share']}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This is the first common-feasible-set release diagnostic. "
                "Distinct release terminals are wall-facing evidence, but still "
                "not a promoted wall/pathway or method-success claim."
            ),
            "",
        ]
    )
    (output_dir / ANCHOR_RELEASE_REPORT_MD).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--boundary-plan-dir", type=Path, default=DEFAULT_BOUNDARY_PLAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ranks", default="4")
    parser.add_argument("--method-seeds", default="0")
    parser.add_argument("--max-targets-per-role", type=int, default=1)
    parser.add_argument("--max-total-pairs", type=int, default=0)
    parser.add_argument(
        "--target-selection-policy",
        choices=sorted(TARGET_SELECTION_POLICIES),
        default="default",
    )
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--release-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
