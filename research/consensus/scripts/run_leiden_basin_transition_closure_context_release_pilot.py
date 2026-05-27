#!/usr/bin/env python3
"""Pilot bounded closure-context release after direct closure shrink.

This diagnostic tests whether direct-node-only refinement failed because the
mutable region was too local. It starts from recreated vanilla, releases a
small closure-context subset for positive direct-shrink prefixes, and measures
quality/support shift. It is not a production Leiden or Dongdaemun policy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from analyze_leiden_basin_transition_boundaries import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_BOUNDARY_DIR,
    NODE_ROWS_FILENAME,
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
)
from rank_leiden_basin_transition_closure_frontier import (  # noqa: E402
    PAIR_COLUMNS,
)
from run_leiden_basin_transition_closure_operator_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_DIRECT_OPERATOR_DIR,
    ROWS_FILENAME as DIRECT_OPERATOR_ROWS_FILENAME,
    _control_rows_for_pair,
    _evaluate_result,
    _pair_mask,
    _safe_float,
    _score_membership,
    assign_nodes_to_nearest_existing_donor_label,
    direct_nodes_for_frontier_row,
)
from run_leiden_basin_transition_operator_pilot import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_VANILLA_DIR,
    VANILLA_ROWS_FILENAME,
    _find_candidate_row,
    _find_vanilla_row,
    _recreate_candidate,
    _run_leiden,
    _safe_int,
    changed_support_nodes,
    fixed_outside,
)
from run_leiden_hysteresis_work_acceleration_monitor import (  # noqa: E402
    _compact_membership,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "basin_transition_closure_context_release_pilot_field34_cc"
)
ROWS_FILENAME = "basin_transition_closure_context_release_rows.csv"
SUMMARY_FILENAME = "basin_transition_closure_context_release_summary.json"
REPORT_FILENAME = "basin_transition_closure_context_release_report.md"

RAW_OPERATOR = "closure_context_release_candidate_nearest_raw"
POLISH_OPERATOR = "closure_context_release_candidate_nearest_direct_polish"
CONTEXT_OPERATOR_NAMES = (RAW_OPERATOR, POLISH_OPERATOR)
DEFAULT_SOURCE_OPERATORS = (
    "closure_split_shrink_from_vanilla_candidate_nearest_raw",
    "closure_split_shrink_from_vanilla_candidate_nearest_direct_polish",
)


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_label_prefix(value: Any) -> tuple[int, ...]:
    text = str(value)
    if not text or text.lower() == "nan":
        return ()
    return tuple(int(part) for part in text.split(",") if part)


def selected_direct_prefix_rows(
    rows: pd.DataFrame,
    *,
    source_operators: tuple[str, ...],
    source_labels: tuple[str, ...],
    max_pairs: int,
    max_prefixes_per_pair: int,
) -> pd.DataFrame:
    """Select positive direct-shrink prefixes for context release."""
    if rows.empty:
        return rows.copy()
    frame = rows[
        rows["operator"].astype(str).isin(source_operators)
        & rows["diagnostic_label"].astype(str).isin(source_labels)
        & pd.to_numeric(rows["step_index"], errors="coerce").gt(0)
    ].copy()
    if frame.empty:
        return frame
    frame["delta_vs_control_extra"] = pd.to_numeric(
        frame["delta_vs_control_extra"],
        errors="coerce",
    )
    frame["support_burden_reduction_vs_vanilla"] = pd.to_numeric(
        frame["support_burden_reduction_vs_vanilla"],
        errors="coerce",
    )
    frame["step_index"] = pd.to_numeric(frame["step_index"], errors="coerce")
    frame = frame.sort_values(
        [
            *PAIR_COLUMNS,
            "delta_vs_control_extra",
            "support_burden_reduction_vs_vanilla",
            "step_index",
        ],
        ascending=[*([True] * len(PAIR_COLUMNS)), False, False, True],
    )
    frame = frame.groupby(PAIR_COLUMNS, dropna=False).head(int(max_prefixes_per_pair))
    pair_keys = frame[PAIR_COLUMNS].drop_duplicates().head(int(max_pairs))
    keep = np.zeros(len(frame), dtype=np.bool_)
    for _, pair in pair_keys.iterrows():
        keep |= _pair_mask(frame, pair)
    return frame[keep].reset_index(drop=True)


def _node_mask(node_count: int, nodes: np.ndarray) -> np.ndarray:
    mask = np.zeros(int(node_count), dtype=np.bool_)
    if nodes.size:
        mask[np.asarray(nodes, dtype=np.int64)] = True
    return mask


def edge_pull_to_direct_nodes(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    candidate_nodes: np.ndarray,
    direct_nodes: np.ndarray,
    node_count: int,
) -> pd.DataFrame:
    """Score context candidates by weighted adjacency to direct edit nodes."""
    candidates = np.asarray(candidate_nodes, dtype=np.uint32)
    if candidates.size == 0:
        return pd.DataFrame(columns=["node", "edge_pull_to_direct"])
    direct_mask = _node_mask(node_count, np.asarray(direct_nodes, dtype=np.uint32))
    candidate_mask = _node_mask(node_count, candidates)
    pulls = {int(node): 0.0 for node in candidates}
    edge_mask = (
        (candidate_mask[src.astype(np.int64)] & direct_mask[dst.astype(np.int64)])
        | (candidate_mask[dst.astype(np.int64)] & direct_mask[src.astype(np.int64)])
    )
    for u_raw, v_raw, w_raw in zip(
        src[edge_mask],
        dst[edge_mask],
        weight[edge_mask],
        strict=False,
    ):
        u = int(u_raw)
        v = int(v_raw)
        w = float(w_raw)
        if u in pulls:
            pulls[u] += w
        if v in pulls:
            pulls[v] += w
    return pd.DataFrame(
        {
            "node": list(pulls.keys()),
            "edge_pull_to_direct": list(pulls.values()),
        }
    )


def bounded_context_nodes_for_label(
    *,
    candidate_membership: np.ndarray,
    label: int,
    direct_nodes: np.ndarray,
    support_union: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_weights: np.ndarray,
    max_context_nodes: int,
    context_pool: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a bounded same-label context subset for one closure label."""
    all_label_nodes = np.flatnonzero(
        np.asarray(candidate_membership, dtype=np.uint64) == np.uint64(int(label))
    ).astype(np.uint32, copy=False)
    direct_set = {int(node) for node in np.asarray(direct_nodes, dtype=np.uint32)}
    support_set = {int(node) for node in np.asarray(support_union, dtype=np.uint32)}
    candidates = [int(node) for node in all_label_nodes if int(node) not in direct_set]
    if context_pool == "outside_support":
        candidates = [node for node in candidates if node not in support_set]
    elif context_pool != "all_context":
        raise ValueError(f"Unsupported context pool: {context_pool}")
    if not candidates or max_context_nodes <= 0:
        return np.asarray([], dtype=np.uint32), {
            "candidate_context_node_count": int(len(candidates)),
            "selected_context_edge_pull_sum": 0.0,
        }
    scored = edge_pull_to_direct_nodes(
        src=src,
        dst=dst,
        weight=weight,
        candidate_nodes=np.asarray(candidates, dtype=np.uint32),
        direct_nodes=direct_nodes,
        node_count=int(candidate_membership.size),
    )
    scored["node_weight"] = [
        float(node_weights[int(node)]) for node in scored["node"].astype(int)
    ]
    scored = scored.sort_values(
        ["edge_pull_to_direct", "node_weight", "node"],
        ascending=[False, False, True],
    )
    selected = scored.head(int(max_context_nodes))
    return np.asarray(selected["node"], dtype=np.uint32), {
        "candidate_context_node_count": int(len(candidates)),
        "selected_context_edge_pull_sum": float(
            selected["edge_pull_to_direct"].sum()
        ),
    }


def _release_stats_zero() -> dict[str, Any]:
    return {
        "source_operator": "",
        "source_step_index": 0,
        "source_delta_vs_vanilla": math.nan,
        "source_delta_vs_control_extra": math.nan,
        "source_support_reduction_vs_vanilla": 0,
        "released_closure_labels": "",
        "released_label_count": 0,
        "released_direct_node_count": 0,
        "released_context_node_count": 0,
        "released_context_candidate_pool_count": 0,
        "released_context_edge_pull_sum": 0.0,
        "context_pool": "",
        "context_budget_per_label": 0,
        "mutable_node_count": 0,
        "target_donor": "",
        "fixed_outside_mutable": False,
    }


def _release_stats_for_prefix(
    *,
    prefix: pd.Series,
    labels: tuple[int, ...],
    direct_nodes: np.ndarray,
    context_nodes: np.ndarray,
    context_candidate_pool_count: int,
    context_edge_pull_sum: float,
    context_pool: str,
    context_budget_per_label: int,
    fixed_outside_mutable: bool,
) -> dict[str, Any]:
    return {
        "source_operator": prefix["operator"],
        "source_step_index": int(prefix["step_index"]),
        "source_delta_vs_vanilla": _safe_float(prefix.get("delta_vs_vanilla")),
        "source_delta_vs_control_extra": _safe_float(
            prefix.get("delta_vs_control_extra")
        ),
        "source_support_reduction_vs_vanilla": int(
            _safe_float(prefix.get("support_burden_reduction_vs_vanilla"), 0.0)
        ),
        "released_closure_labels": ",".join(str(label) for label in labels),
        "released_label_count": int(len(labels)),
        "released_direct_node_count": int(direct_nodes.size),
        "released_context_node_count": int(context_nodes.size),
        "released_context_candidate_pool_count": int(context_candidate_pool_count),
        "released_context_edge_pull_sum": float(context_edge_pull_sum),
        "context_pool": context_pool,
        "context_budget_per_label": int(context_budget_per_label),
        "mutable_node_count": int(len(set(direct_nodes) | set(context_nodes))),
        "target_donor": "candidate_nearest_existing",
        "fixed_outside_mutable": bool(fixed_outside_mutable),
    }


def diagnostic_label_for_context_row(
    row: pd.Series,
    *,
    material_delta: float = 1e-9,
    min_support_shift_from_vanilla: float = 0.1,
) -> str:
    if str(row["operator"]) not in CONTEXT_OPERATOR_NAMES:
        return "control"
    if float(row["delta_vs_vanilla"]) < -float(material_delta):
        return "quality_loss"
    if float(row["delta_vs_control_extra"]) < -float(material_delta):
        return "seed_control_dominates"
    if (
        float(row["result_support_distance_to_vanilla"])
        < float(min_support_shift_from_vanilla)
    ):
        return "quality_win_same_basin"
    return "quality_win_support_shift"


def _operator_rows_for_prefix(
    *,
    graph: Any,
    arrays: Any,
    node_weights: np.ndarray,
    baseline: Any,
    candidate: Any,
    vanilla: Any,
    prefix: pd.Series,
    node_rows: pd.DataFrame,
    candidate_support: np.ndarray,
    vanilla_support: np.ndarray,
    sketch_nodes: np.ndarray,
    context: dict[str, Any],
    resolution: float,
    randomness: float,
    local_polish_iterations: int,
    operator_seed: int,
    context_budget_per_label: int,
    context_pool: str,
) -> list[dict[str, Any]]:
    labels = parse_label_prefix(prefix["released_closure_labels"])
    if not labels:
        return []
    support_union = np.union1d(
        np.asarray(candidate_support, dtype=np.uint32),
        np.asarray(vanilla_support, dtype=np.uint32),
    )
    direct_by_label: list[np.ndarray] = []
    context_by_label: list[np.ndarray] = []
    context_candidate_pool_count = 0
    context_edge_pull_sum = 0.0
    for label in labels:
        frontier_like = pd.Series(
            {
                **{column: prefix[column] for column in PAIR_COLUMNS},
                "closure_mode": prefix["closure_mode"],
                "closure_label": label,
            }
        )
        direct = direct_nodes_for_frontier_row(
            node_rows=node_rows,
            frontier_row=frontier_like,
        )
        selected_context, stats = bounded_context_nodes_for_label(
            candidate_membership=candidate.recreated.membership,
            label=label,
            direct_nodes=direct,
            support_union=support_union,
            src=arrays.src,
            dst=arrays.dst,
            weight=arrays.weight,
            node_weights=node_weights,
            max_context_nodes=context_budget_per_label,
            context_pool=context_pool,
        )
        direct_by_label.append(direct)
        context_by_label.append(selected_context)
        context_candidate_pool_count += int(stats["candidate_context_node_count"])
        context_edge_pull_sum += float(stats["selected_context_edge_pull_sum"])
    direct_nodes = (
        np.unique(np.concatenate(direct_by_label)).astype(np.uint32, copy=False)
        if direct_by_label
        else np.asarray([], dtype=np.uint32)
    )
    context_nodes = (
        np.unique(np.concatenate(context_by_label)).astype(np.uint32, copy=False)
        if context_by_label
        else np.asarray([], dtype=np.uint32)
    )
    mutable_nodes = np.unique(np.concatenate([direct_nodes, context_nodes])).astype(
        np.uint32,
        copy=False,
    )
    if mutable_nodes.size == 0:
        return []
    edited, _next_label = assign_nodes_to_nearest_existing_donor_label(
        vanilla.membership,
        candidate.recreated.membership,
        mutable_nodes,
        blocked_nodes=mutable_nodes,
    )
    edited = _compact_membership(edited)
    raw_stats = _release_stats_for_prefix(
        prefix=prefix,
        labels=labels,
        direct_nodes=direct_nodes,
        context_nodes=context_nodes,
        context_candidate_pool_count=context_candidate_pool_count,
        context_edge_pull_sum=context_edge_pull_sum,
        context_pool=context_pool,
        context_budget_per_label=context_budget_per_label,
        fixed_outside_mutable=False,
    )
    raw = _score_membership(graph, edited, resolution=resolution)
    rows = [
        _evaluate_result(
            context=context,
            operator=RAW_OPERATOR,
            result=raw,
            baseline=baseline,
            candidate=candidate,
            vanilla=vanilla,
            candidate_support=candidate_support,
            vanilla_support=vanilla_support,
            sketch_nodes=sketch_nodes,
            released_stats=raw_stats,
        )
    ]
    if local_polish_iterations <= 0:
        return rows
    fixed = fixed_outside(int(edited.size), mutable_nodes)
    polished = _run_leiden(
        graph,
        resolution=resolution,
        seed=operator_seed + int(prefix["step_index"]),
        n_iterations=local_polish_iterations,
        randomness=randomness,
        initial_membership=edited,
        fixed_nodes=fixed,
    )
    polish_stats = {**raw_stats, "fixed_outside_mutable": True}
    rows.append(
        _evaluate_result(
            context=context,
            operator=POLISH_OPERATOR,
            result=polished,
            baseline=baseline,
            candidate=candidate,
            vanilla=vanilla,
            candidate_support=candidate_support,
            vanilla_support=vanilla_support,
            sketch_nodes=sketch_nodes,
            released_stats=polish_stats,
        )
    )
    return rows


def run_pilot(
    *,
    direct_operator_dir: Path,
    boundary_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    output_dir: Path,
    source_operators: tuple[str, ...],
    source_labels: tuple[str, ...],
    max_pairs: int,
    max_prefixes_per_pair: int,
    context_budget_per_label: int,
    context_pool: str,
    baseline_iterations: int,
    transition_iterations: int,
    polish_iterations: int,
    local_polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_rows = pd.read_csv(direct_operator_dir / DIRECT_OPERATOR_ROWS_FILENAME)
    selected_prefixes = selected_direct_prefix_rows(
        direct_rows,
        source_operators=source_operators,
        source_labels=source_labels,
        max_pairs=max_pairs,
        max_prefixes_per_pair=max_prefixes_per_pair,
    )
    if selected_prefixes.empty:
        raise ValueError("No positive direct-shrink prefixes selected")
    node_rows = pd.read_csv(boundary_dir / NODE_ROWS_FILENAME)
    candidates = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)

    baseline_cache: dict[str, Any] = {}
    candidate_cache: dict[tuple[str, int], Any] = {}
    vanilla_cache: dict[tuple[str, int, float, str], Any] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, Any]] = {}
    out_rows: list[dict[str, Any]] = []
    pair_rows = selected_prefixes[PAIR_COLUMNS].drop_duplicates().reset_index(drop=True)

    for _, pair in pair_rows.iterrows():
        case = str(pair["case"])
        candidate_index = int(pair["candidate_index"])
        seed = int(pair["vanilla_seed"])
        vanilla_randomness = float(pair["vanilla_randomness"])
        vanilla_n = str(pair["vanilla_requested_n_iterations"])
        candidate_row = _find_candidate_row(
            candidates,
            case=case,
            candidate_index=candidate_index,
        )
        vanilla_row = _find_vanilla_row(
            vanilla_rows,
            case=case,
            seed=seed,
            randomness=vanilla_randomness,
            n_iterations=vanilla_n,
        )
        graph_dir = Path(str(vanilla_row["graph_dir"]))
        graph_key = str(graph_dir)
        if graph_key not in graph_cache:
            graph_cache[graph_key] = _load_graph(graph_dir)
        graph, node_weights, arrays = graph_cache[graph_key]
        if case not in baseline_cache:
            baseline_cache[case] = _run_leiden(
                graph,
                resolution=resolution,
                seed=int(candidate_row.get("seed", 0)),
                n_iterations=baseline_iterations,
                randomness=randomness,
            )
        baseline = baseline_cache[case]
        ckey = (case, candidate_index)
        if ckey not in candidate_cache:
            candidate_cache[ckey] = _recreate_candidate(
                graph=graph,
                arrays=arrays,
                node_weights=node_weights,
                baseline_membership=baseline.membership,
                baseline_quality=baseline.quality,
                row=candidate_row,
                resolution=resolution,
                randomness=randomness,
                perturb_seed_offset=perturb_seed_offset,
                polish_iterations=polish_iterations,
            )
        candidate = candidate_cache[ckey]
        vkey = (case, seed, vanilla_randomness, vanilla_n)
        if vkey not in vanilla_cache:
            vanilla_cache[vkey] = _run_leiden(
                graph,
                resolution=resolution,
                seed=seed,
                n_iterations=int(
                    _safe_int(vanilla_n, baseline_iterations) or baseline_iterations
                ),
                randomness=vanilla_randomness,
            )
        vanilla = vanilla_cache[vkey]
        sketch_nodes, sketch_context = compatible_sketch_nodes(
            arrays=arrays,
            baseline_membership=baseline.membership,
            node_weights=node_weights,
            candidate_rows=candidates[candidates["case"].astype(str) == case],
        )
        if not bool(sketch_context.get("sketch_context_hash_matches_candidate", False)):
            raise RuntimeError(f"sketch context mismatch for {case}")
        candidate_support = candidate.support_nodes
        vanilla_support = changed_support_nodes(baseline.membership, vanilla.membership)
        operator_seed = (
            int(candidate_row.get("seed", 0))
            + int(perturb_seed_offset)
            + int(candidate_index)
        )
        context = {
            "case": case,
            "field": pair["field"],
            "method": pair["method"],
            "candidate_index": candidate_index,
            "vanilla_seed": seed,
            "vanilla_randomness": vanilla_randomness,
            "vanilla_requested_n_iterations": vanilla_n,
            "closure_mode": selected_prefixes[_pair_mask(selected_prefixes, pair)][
                "closure_mode"
            ].iloc[0],
            "baseline_iterations": int(baseline_iterations),
            "transition_iterations": int(transition_iterations),
            "polish_iterations": int(polish_iterations),
            "local_polish_iterations": int(local_polish_iterations),
            "resolution": float(resolution),
            "randomness": float(randomness),
        }
        out_rows.extend(
            _control_rows_for_pair(
                graph=graph,
                baseline=baseline,
                candidate=candidate,
                vanilla=vanilla,
                candidate_support=candidate_support,
                vanilla_support=vanilla_support,
                sketch_nodes=sketch_nodes,
                context=context,
                resolution=resolution,
                randomness=randomness,
                transition_iterations=transition_iterations,
                operator_seed=operator_seed,
            )
        )
        for _, prefix in selected_prefixes[_pair_mask(selected_prefixes, pair)].iterrows():
            out_rows.extend(
                _operator_rows_for_prefix(
                    graph=graph,
                    arrays=arrays,
                    node_weights=node_weights,
                    baseline=baseline,
                    candidate=candidate,
                    vanilla=vanilla,
                    prefix=prefix,
                    node_rows=node_rows,
                    candidate_support=candidate_support,
                    vanilla_support=vanilla_support,
                    sketch_nodes=sketch_nodes,
                    context=context,
                    resolution=resolution,
                    randomness=randomness,
                    local_polish_iterations=local_polish_iterations,
                    operator_seed=operator_seed,
                    context_budget_per_label=context_budget_per_label,
                    context_pool=context_pool,
                )
            )

    rows = pd.DataFrame(out_rows)
    if not rows.empty:
        control_quality = rows[
            rows["operator"].eq("control_extra_from_baseline")
        ][[*PAIR_COLUMNS, "quality"]].rename(
            columns={"quality": "control_extra_quality"}
        )
        rows = rows.merge(control_quality, on=PAIR_COLUMNS, how="left")
        rows["delta_vs_control_extra"] = (
            rows["quality"] - rows["control_extra_quality"]
        )
        rows["diagnostic_label"] = [
            diagnostic_label_for_context_row(row) for _, row in rows.iterrows()
        ]
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    context_rows = rows[rows["operator"].isin(CONTEXT_OPERATOR_NAMES)]
    summary = {
        "schema": "leiden_basin_transition_closure_context_release_pilot.v1",
        "direct_operator_dir": str(direct_operator_dir),
        "boundary_dir": str(boundary_dir),
        "vanilla_dir": str(vanilla_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "output_dir": str(output_dir),
        "source_operators": list(source_operators),
        "source_labels": list(source_labels),
        "selected_pair_count": int(len(pair_rows)),
        "selected_prefix_rows": int(len(selected_prefixes)),
        "operator_rows": int(len(rows)),
        "context_operator_rows": int(len(context_rows)),
        "context_budget_per_label": int(context_budget_per_label),
        "context_pool": context_pool,
        "baseline_iterations": int(baseline_iterations),
        "transition_iterations": int(transition_iterations),
        "polish_iterations": int(polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
    }
    if not context_rows.empty:
        summary["best_context_delta_vs_vanilla"] = float(
            context_rows["delta_vs_vanilla"].max()
        )
        summary["best_context_delta_vs_control_extra"] = float(
            context_rows["delta_vs_control_extra"].max()
        )
        summary["max_context_support_shift_from_vanilla"] = float(
            context_rows["result_support_distance_to_vanilla"].max()
        )
        summary["min_context_support_distance_to_candidate"] = float(
            context_rows["result_support_distance_to_candidate"].min()
        )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / REPORT_FILENAME, rows, summary)
    return summary


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 32) -> list[str]:
    if frame.empty:
        return []
    display = frame.head(max_rows)
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(path: Path, rows: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [
        "# Closure Context Release Pilot",
        "",
        "This diagnostic tests bounded same-label closure-context release after positive direct shrink prefixes.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "selected_pair_count",
        "selected_prefix_rows",
        "operator_rows",
        "context_operator_rows",
        "best_context_delta_vs_vanilla",
        "best_context_delta_vs_control_extra",
        "max_context_support_shift_from_vanilla",
        "min_context_support_distance_to_candidate",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Operator Summary", ""])
    if not rows.empty:
        operator_summary = (
            rows.groupby("operator", as_index=False)
            .agg(
                rows=("operator", "size"),
                delta_vs_vanilla_max=("delta_vs_vanilla", "max"),
                delta_vs_vanilla_median=("delta_vs_vanilla", "median"),
                delta_vs_control_extra_max=("delta_vs_control_extra", "max"),
                delta_vs_control_extra_median=("delta_vs_control_extra", "median"),
                support_shift_from_vanilla_max=(
                    "result_support_distance_to_vanilla",
                    "max",
                ),
                support_shift_from_vanilla_median=(
                    "result_support_distance_to_vanilla",
                    "median",
                ),
                support_distance_to_candidate_median=(
                    "result_support_distance_to_candidate",
                    "median",
                ),
                endpoint_distance_to_candidate_median=(
                    "result_endpoint_distance_to_candidate",
                    "median",
                ),
                released_context_nodes_median=(
                    "released_context_node_count",
                    "median",
                ),
                elapsed_sec_median=("elapsed_sec", "median"),
            )
            .sort_values("operator")
        )
        lines.extend(_markdown_table(operator_summary, max_rows=40))
    lines.extend(["", "## Context Rows", ""])
    display_cols = [
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "operator",
        "source_operator",
        "source_step_index",
        "released_label_count",
        "released_direct_node_count",
        "released_context_node_count",
        "delta_vs_vanilla",
        "delta_vs_control_extra",
        "diagnostic_label",
        "result_support_distance_to_vanilla",
        "result_support_distance_to_candidate",
        "result_endpoint_distance_to_candidate",
        "elapsed_sec",
    ]
    context_rows = rows[rows["operator"].isin(CONTEXT_OPERATOR_NAMES)]
    lines.extend(
        _markdown_table(
            context_rows[[c for c in display_cols if c in context_rows.columns]],
            max_rows=60,
        )
    )
    lines.extend(["", "## Diagnostic Labels", ""])
    if not context_rows.empty:
        labels = (
            context_rows.groupby(["operator", "diagnostic_label"], as_index=False)
            .agg(rows=("operator", "size"))
            .sort_values(["operator", "diagnostic_label"])
        )
        lines.extend(_markdown_table(labels, max_rows=60))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- This is a bounded refinement-distance diagnostic, not a policy sweep.",
            "- `quality_win_support_shift` requires quality to beat vanilla/control and support to move away from vanilla.",
            "- If context release remains `quality_win_same_basin`, the shrink family should stop.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct-operator-dir",
        type=Path,
        default=DEFAULT_DIRECT_OPERATOR_DIR,
    )
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-operators",
        default=",".join(DEFAULT_SOURCE_OPERATORS),
    )
    parser.add_argument("--source-labels", default="quality_win_same_basin")
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--max-prefixes-per-pair", type=int, default=2)
    parser.add_argument("--context-budget-per-label", type=int, default=16)
    parser.add_argument(
        "--context-pool",
        choices=("outside_support", "all_context"),
        default="outside_support",
    )
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--transition-iterations", type=int, default=5)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--local-polish-iterations", type=int, default=3)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_pilot(
        direct_operator_dir=args.direct_operator_dir,
        boundary_dir=args.boundary_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dir=args.vanilla_dir,
        output_dir=args.output_dir,
        source_operators=_parse_csv_tuple(args.source_operators),
        source_labels=_parse_csv_tuple(args.source_labels),
        max_pairs=args.max_pairs,
        max_prefixes_per_pair=args.max_prefixes_per_pair,
        context_budget_per_label=args.context_budget_per_label,
        context_pool=args.context_pool,
        baseline_iterations=args.baseline_iterations,
        transition_iterations=args.transition_iterations,
        polish_iterations=args.polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
