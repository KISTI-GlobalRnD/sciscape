#!/usr/bin/env python3
"""Analyze boundary structure for endpoint-near basin transitions.

This is diagnostic-only. It recomputes full baseline/candidate/vanilla
memberships for selected candidate-vs-vanilla pairs, then explains the full
changed-support boundary before any shrink/expand operator mutates membership.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
)
from run_leiden_basin_transition_operator_pilot import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_LANDSCAPE_DIR,
    DEFAULT_VANILLA_DIR,
    HYPOTHESES_FILENAME,
    VANILLA_ROWS_FILENAME,
    _find_candidate_row,
    _find_vanilla_row,
    _parse_candidate_index,
    _parse_vanilla_config,
    _recreate_candidate,
    _run_leiden,
    _safe_int,
    _select_hypotheses,
    changed_support_nodes,
    endpoint_distance,
    support_distance,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "basin_transition_boundary_analysis_field34_cc"
)
NODE_ROWS_FILENAME = "basin_transition_boundary_node_rows.csv"
GROUP_ROWS_FILENAME = "basin_transition_boundary_group_rows.csv"
PAIR_ROWS_FILENAME = "basin_transition_boundary_pair_rows.csv"
SUMMARY_FILENAME = "basin_transition_boundary_summary.json"
REPORT_FILENAME = "basin_transition_boundary_report.md"

def _as_node_array(values: np.ndarray | list[int] | set[int]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        out = values.astype(np.uint32, copy=False)
    else:
        out = np.asarray(sorted(int(value) for value in values), dtype=np.uint32)
    if out.size == 0:
        return out
    return np.unique(out)

def classify_support_nodes(
    candidate_support: np.ndarray | list[int] | set[int],
    vanilla_support: np.ndarray | list[int] | set[int],
) -> dict[int, str]:
    """Classify changed-support nodes using candidate and vanilla footprints."""
    candidate = {int(node) for node in _as_node_array(candidate_support)}
    vanilla = {int(node) for node in _as_node_array(vanilla_support)}
    classes: dict[int, str] = {}
    for node in sorted(candidate | vanilla):
        if node in candidate and node in vanilla:
            classes[node] = "shared"
        elif node in candidate:
            classes[node] = "core"
        else:
            classes[node] = "vanilla_extra"
    return classes

def boundary_anchor_nodes(
    src: np.ndarray,
    dst: np.ndarray,
    support_nodes: np.ndarray | list[int] | set[int],
    *,
    max_anchors: int,
) -> tuple[np.ndarray, int]:
    """Return one-hop non-support boundary anchors and the truncated count."""
    support = _as_node_array(support_nodes)
    if support.size == 0 or max_anchors <= 0:
        return np.asarray([], dtype=np.uint32), 0
    max_node = int(
        max(
            int(np.max(src, initial=0)),
            int(np.max(dst, initial=0)),
            int(np.max(support)),
        )
    )
    support_mask = np.zeros(max_node + 1, dtype=np.bool_)
    support_mask[support.astype(np.int64)] = True
    incident = support_mask[src.astype(np.int64)] | support_mask[dst.astype(np.int64)]
    neighbors = np.concatenate(
        [
            src[incident].astype(np.uint32, copy=False),
            dst[incident].astype(np.uint32, copy=False),
        ]
    )
    anchors = np.setdiff1d(np.unique(neighbors), support, assume_unique=False)
    truncated = max(0, int(anchors.size) - int(max_anchors))
    return anchors[: int(max_anchors)].astype(np.uint32, copy=False), truncated

def _new_metric() -> dict[str, Any]:
    return {
        "incident_weight_total": 0.0,
        "edge_weight_to_candidate_support": 0.0,
        "edge_weight_to_vanilla_extra": 0.0,
        "edge_weight_to_same_baseline_label": 0.0,
        "edge_weight_to_candidate_label": 0.0,
        "edge_weight_to_vanilla_label": 0.0,
        "_candidate_label_weights": {},
        "_vanilla_label_weights": {},
    }

def _add_label_weight(weights: dict[int, float], label: int, value: float) -> None:
    weights[label] = float(weights.get(label, 0.0) + value)

def _strongest_label(weights: dict[int, float]) -> tuple[int | None, float]:
    if not weights:
        return None, 0.0
    label, value = max(weights.items(), key=lambda item: (item[1], -item[0]))
    return int(label), float(value)

def _ratio(value: float, total: float) -> float:
    if not math.isfinite(total) or total <= 0.0:
        return 0.0
    return float(value) / float(total)

def classify_boundary_role(
    *,
    support_class: str,
    bridge_score: float,
    collateral_score: float,
    role_margin: float,
) -> str:
    if support_class == "core":
        return "candidate_core"
    if support_class == "shared":
        return "shared"
    if support_class == "boundary_anchor":
        return "anchor"
    if support_class != "vanilla_extra":
        return "unknown"
    diff = float(bridge_score) - float(collateral_score)
    if abs(diff) <= float(role_margin):
        return "ambiguous"
    if diff > 0.0:
        return "bridge_like"
    return "collateral_like"

def compute_boundary_node_rows(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_weights: np.ndarray,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    candidate_support: np.ndarray,
    vanilla_support: np.ndarray,
    context: dict[str, Any],
    max_boundary_anchors: int,
    role_margin: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute full-support boundary rows for one candidate-vs-vanilla pair."""
    candidate_support = _as_node_array(candidate_support)
    vanilla_support = _as_node_array(vanilla_support)
    support_classes = classify_support_nodes(candidate_support, vanilla_support)
    support_union = _as_node_array(set(support_classes))
    anchors, truncated_anchors = boundary_anchor_nodes(
        src,
        dst,
        support_union,
        max_anchors=max_boundary_anchors,
    )
    for node in anchors:
        support_classes[int(node)] = "boundary_anchor"

    selected = _as_node_array(set(support_classes))
    metrics = {int(node): _new_metric() for node in selected}
    if selected.size:
        node_count = int(max(len(baseline_membership), int(np.max(selected)) + 1))
        selected_mask = np.zeros(node_count, dtype=np.bool_)
        selected_mask[selected.astype(np.int64)] = True
        candidate_support_mask = np.zeros(node_count, dtype=np.bool_)
        candidate_support_mask[candidate_support.astype(np.int64)] = True
        vanilla_extra_nodes = np.asarray(
            [
                node
                for node, klass in support_classes.items()
                if klass == "vanilla_extra"
            ],
            dtype=np.uint32,
        )
        vanilla_extra_mask = np.zeros(node_count, dtype=np.bool_)
        if vanilla_extra_nodes.size:
            vanilla_extra_mask[vanilla_extra_nodes.astype(np.int64)] = True
        edge_mask = selected_mask[src.astype(np.int64)] | selected_mask[
            dst.astype(np.int64)
        ]
        for u_raw, v_raw, w_raw in zip(
            src[edge_mask],
            dst[edge_mask],
            weight[edge_mask],
            strict=False,
        ):
            u = int(u_raw)
            v = int(v_raw)
            w = float(w_raw)
            if u in metrics:
                _update_node_metric(
                    metrics[u],
                    node=u,
                    neighbor=v,
                    edge_weight=w,
                    baseline_membership=baseline_membership,
                    candidate_membership=candidate_membership,
                    vanilla_membership=vanilla_membership,
                    candidate_support_mask=candidate_support_mask,
                    vanilla_extra_mask=vanilla_extra_mask,
                )
            if v != u and v in metrics:
                _update_node_metric(
                    metrics[v],
                    node=v,
                    neighbor=u,
                    edge_weight=w,
                    baseline_membership=baseline_membership,
                    candidate_membership=candidate_membership,
                    vanilla_membership=vanilla_membership,
                    candidate_support_mask=candidate_support_mask,
                    vanilla_extra_mask=vanilla_extra_mask,
                )

    rows: list[dict[str, Any]] = []
    for node in selected:
        node_int = int(node)
        metric = metrics[node_int]
        total = float(metric["incident_weight_total"])
        core_pull = _ratio(metric["edge_weight_to_candidate_support"], total)
        extra_pull = _ratio(metric["edge_weight_to_vanilla_extra"], total)
        baseline_pull = _ratio(metric["edge_weight_to_same_baseline_label"], total)
        candidate_pull = _ratio(metric["edge_weight_to_candidate_label"], total)
        vanilla_pull = _ratio(metric["edge_weight_to_vanilla_label"], total)
        bridge_score = core_pull * vanilla_pull
        collateral_score = baseline_pull * (1.0 - core_pull)
        keep_vanilla_score = vanilla_pull
        revert_baseline_score = baseline_pull
        candidate_nearest_score = candidate_pull
        necessity_score = keep_vanilla_score - max(
            revert_baseline_score,
            candidate_nearest_score,
        )
        candidate_label, candidate_label_weight = _strongest_label(
            metric["_candidate_label_weights"]
        )
        vanilla_label, vanilla_label_weight = _strongest_label(
            metric["_vanilla_label_weights"]
        )
        support_class = str(support_classes[node_int])
        rows.append(
            {
                **context,
                "node": node_int,
                "support_class": support_class,
                "boundary_role": classify_boundary_role(
                    support_class=support_class,
                    bridge_score=bridge_score,
                    collateral_score=collateral_score,
                    role_margin=role_margin,
                ),
                "baseline_label": int(baseline_membership[node_int]),
                "candidate_label": int(candidate_membership[node_int]),
                "vanilla_label": int(vanilla_membership[node_int]),
                "node_weight": float(node_weights[node_int]),
                "incident_weight_total": total,
                "edge_weight_to_candidate_support": float(
                    metric["edge_weight_to_candidate_support"]
                ),
                "edge_weight_to_vanilla_extra": float(
                    metric["edge_weight_to_vanilla_extra"]
                ),
                "edge_weight_to_same_baseline_label": float(
                    metric["edge_weight_to_same_baseline_label"]
                ),
                "edge_weight_to_candidate_label": float(
                    metric["edge_weight_to_candidate_label"]
                ),
                "edge_weight_to_vanilla_label": float(
                    metric["edge_weight_to_vanilla_label"]
                ),
                "strongest_candidate_pull_label": candidate_label,
                "strongest_candidate_pull_weight": candidate_label_weight,
                "strongest_vanilla_pull_label": vanilla_label,
                "strongest_vanilla_pull_weight": vanilla_label_weight,
                "core_pull": core_pull,
                "vanilla_extra_pull": extra_pull,
                "baseline_pull": baseline_pull,
                "candidate_pull": candidate_pull,
                "vanilla_pull": vanilla_pull,
                "bridge_score": bridge_score,
                "collateral_score": collateral_score,
                "necessity_score": necessity_score,
                "keep_vanilla_proxy_score": keep_vanilla_score,
                "revert_baseline_proxy_score": revert_baseline_score,
                "candidate_nearest_proxy_score": candidate_nearest_score,
                "boundary_role_margin": bridge_score - collateral_score,
            }
        )
    summary = {
        "candidate_support_size": int(candidate_support.size),
        "vanilla_support_size": int(vanilla_support.size),
        "support_union_size": int(support_union.size),
        "boundary_anchor_rows": int(anchors.size),
        "boundary_anchor_truncated": int(truncated_anchors),
    }
    return pd.DataFrame(rows), summary

def _update_node_metric(
    metric: dict[str, Any],
    *,
    node: int,
    neighbor: int,
    edge_weight: float,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    candidate_support_mask: np.ndarray,
    vanilla_extra_mask: np.ndarray,
) -> None:
    metric["incident_weight_total"] += edge_weight
    if candidate_support_mask[neighbor]:
        metric["edge_weight_to_candidate_support"] += edge_weight
    if vanilla_extra_mask[neighbor]:
        metric["edge_weight_to_vanilla_extra"] += edge_weight
    if baseline_membership[neighbor] == baseline_membership[node]:
        metric["edge_weight_to_same_baseline_label"] += edge_weight
    if candidate_membership[neighbor] == candidate_membership[node]:
        metric["edge_weight_to_candidate_label"] += edge_weight
    if vanilla_membership[neighbor] == vanilla_membership[node]:
        metric["edge_weight_to_vanilla_label"] += edge_weight
    _add_label_weight(
        metric["_candidate_label_weights"],
        int(candidate_membership[neighbor]),
        edge_weight,
    )
    _add_label_weight(
        metric["_vanilla_label_weights"],
        int(vanilla_membership[neighbor]),
        edge_weight,
    )

def aggregate_boundary_group_rows(node_rows: pd.DataFrame) -> pd.DataFrame:
    if node_rows.empty:
        return pd.DataFrame()
    group_cols = [
        "case",
        "field",
        "method",
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "vanilla_requested_n_iterations",
        "support_class",
        "boundary_role",
        "baseline_label",
        "candidate_label",
        "vanilla_label",
    ]
    aggregations = {
        "node": "size",
        "node_weight": "sum",
        "incident_weight_total": "sum",
        "edge_weight_to_candidate_support": "sum",
        "edge_weight_to_vanilla_extra": "sum",
        "edge_weight_to_same_baseline_label": "sum",
        "edge_weight_to_candidate_label": "sum",
        "edge_weight_to_vanilla_label": "sum",
        "core_pull": "mean",
        "vanilla_extra_pull": "mean",
        "baseline_pull": "mean",
        "candidate_pull": "mean",
        "vanilla_pull": "mean",
        "bridge_score": "mean",
        "collateral_score": "mean",
        "necessity_score": "mean",
        "boundary_role_margin": "mean",
    }
    grouped = (
        node_rows.groupby(group_cols, dropna=False, as_index=False)
        .agg(aggregations)
        .rename(
            columns={
                "node": "node_count",
                "node_weight": "node_weight_sum",
                "core_pull": "core_pull_mean",
                "vanilla_extra_pull": "vanilla_extra_pull_mean",
                "baseline_pull": "baseline_pull_mean",
                "candidate_pull": "candidate_pull_mean",
                "vanilla_pull": "vanilla_pull_mean",
                "bridge_score": "bridge_score_mean",
                "collateral_score": "collateral_score_mean",
                "necessity_score": "necessity_score_mean",
                "boundary_role_margin": "boundary_role_margin_mean",
            }
        )
    )
    return grouped.sort_values(
        [
            "case",
            "candidate_index",
            "vanilla_seed",
            "support_class",
            "boundary_role",
            "node_count",
        ],
        ascending=[True, True, True, True, True, False],
    ).reset_index(drop=True)

def _pair_context(
    *,
    hypothesis: pd.Series,
    candidate_row: pd.Series,
    vanilla_row: pd.Series,
    candidate_index: int,
) -> dict[str, Any]:
    return {
        "case": hypothesis.get("case"),
        "field": hypothesis.get("field"),
        "method": hypothesis.get("method"),
        "candidate_index": int(candidate_index),
        "vanilla_seed": int(vanilla_row["seed"]),
        "vanilla_randomness": float(vanilla_row["randomness"]),
        "vanilla_requested_n_iterations": vanilla_row["requested_n_iterations"],
        "source_cluster": int(candidate_row["source_cluster"]),
        "target_cluster": int(candidate_row["target_cluster"]),
    }

def _pair_summary_row(
    *,
    context: dict[str, Any],
    baseline_quality: float,
    candidate_quality: float,
    vanilla_quality: float,
    candidate_support: np.ndarray,
    vanilla_support: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    sketch_nodes: np.ndarray,
    boundary_summary: dict[str, Any],
) -> dict[str, Any]:
    support_dist, support_intersection, support_union = support_distance(
        candidate_support,
        vanilla_support,
    )
    return {
        **context,
        "baseline_quality": float(baseline_quality),
        "candidate_quality": float(candidate_quality),
        "vanilla_quality": float(vanilla_quality),
        "candidate_delta_vs_baseline": float(candidate_quality - baseline_quality),
        "vanilla_delta_vs_baseline": float(vanilla_quality - baseline_quality),
        "vanilla_minus_candidate_delta": float(vanilla_quality - candidate_quality),
        "candidate_support_size": int(candidate_support.size),
        "vanilla_support_size": int(vanilla_support.size),
        "support_intersection_size": int(support_intersection),
        "support_union_size": int(support_union),
        "support_distance": float(support_dist),
        "endpoint_distance": endpoint_distance(
            candidate_membership,
            vanilla_membership,
            sketch_nodes,
        ),
        **boundary_summary,
    }

def run_analysis(
    *,
    candidate_dirs: tuple[Path, ...],
    landscape_dir: Path,
    vanilla_dir: Path,
    output_dir: Path,
    max_pairs: int,
    candidate_indices: set[int],
    baseline_iterations: int,
    polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    max_boundary_anchors: int,
    role_margin: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hypotheses = pd.read_csv(landscape_dir / HYPOTHESES_FILENAME)
    selected = _select_hypotheses(
        hypotheses,
        max_pairs=max_pairs,
        candidate_indices=candidate_indices,
    )
    if selected.empty:
        raise ValueError("No transition hypotheses selected")
    candidates = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)

    baseline_cache: dict[str, Any] = {}
    candidate_cache: dict[tuple[str, int], Any] = {}
    vanilla_cache: dict[tuple[str, int, float, str], Any] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, Any]] = {}
    node_frames: list[pd.DataFrame] = []
    pair_rows: list[dict[str, Any]] = []

    for _, hypothesis in selected.iterrows():
        case = str(hypothesis["case"])
        candidate_index = _parse_candidate_index(hypothesis["candidate_node_id"])
        seed, vanilla_randomness, vanilla_n = _parse_vanilla_config(
            hypothesis["vanilla_node_id"]
        )
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
                    _safe_int(vanilla_n, baseline_iterations)
                    or baseline_iterations
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

        context = _pair_context(
            hypothesis=hypothesis,
            candidate_row=candidate_row,
            vanilla_row=vanilla_row,
            candidate_index=candidate_index,
        )
        vanilla_support = changed_support_nodes(baseline.membership, vanilla.membership)
        node_rows, boundary_summary = compute_boundary_node_rows(
            src=arrays.src,
            dst=arrays.dst,
            weight=arrays.weight,
            node_weights=node_weights,
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
            candidate_support=candidate.support_nodes,
            vanilla_support=vanilla_support,
            context=context,
            max_boundary_anchors=max_boundary_anchors,
            role_margin=role_margin,
        )
        node_frames.append(node_rows)
        pair_rows.append(
            _pair_summary_row(
                context=context,
                baseline_quality=baseline.quality,
                candidate_quality=candidate.recreated.quality,
                vanilla_quality=vanilla.quality,
                candidate_support=candidate.support_nodes,
                vanilla_support=vanilla_support,
                candidate_membership=candidate.recreated.membership,
                vanilla_membership=vanilla.membership,
                sketch_nodes=sketch_nodes,
                boundary_summary=boundary_summary,
            )
        )

    node_rows = (
        pd.concat(node_frames, ignore_index=True)
        if node_frames
        else pd.DataFrame()
    )
    group_rows = aggregate_boundary_group_rows(node_rows)
    pair_frame = pd.DataFrame(pair_rows)
    node_rows.to_csv(output_dir / NODE_ROWS_FILENAME, index=False)
    group_rows.to_csv(output_dir / GROUP_ROWS_FILENAME, index=False)
    pair_frame.to_csv(output_dir / PAIR_ROWS_FILENAME, index=False)
    summary = {
        "schema": "leiden_basin_transition_boundary_analysis.v1",
        "landscape_dir": str(landscape_dir),
        "vanilla_dir": str(vanilla_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "selected_hypothesis_rows": int(len(selected)),
        "node_rows": int(len(node_rows)),
        "group_rows": int(len(group_rows)),
        "pair_rows": int(len(pair_frame)),
        "baseline_iterations": int(baseline_iterations),
        "polish_iterations": int(polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "max_boundary_anchors": int(max_boundary_anchors),
        "role_margin": float(role_margin),
        "output_dir": str(output_dir),
    }
    if not node_rows.empty:
        summary["support_class_counts"] = {
            str(key): int(value)
            for key, value in node_rows["support_class"].value_counts().items()
        }
        summary["boundary_role_counts"] = {
            str(key): int(value)
            for key, value in node_rows["boundary_role"].value_counts().items()
        }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / REPORT_FILENAME, pair_frame, node_rows, group_rows)
    return summary

def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> list[str]:
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

def write_report(
    path: Path,
    pair_rows: pd.DataFrame,
    node_rows: pd.DataFrame,
    group_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Basin Transition Boundary Analysis",
        "",
        "This is a diagnostic dry run. It classifies changed-support boundaries before any shrink/expand operator mutates membership.",
        "",
        "## Pair Summary",
        "",
    ]
    pair_cols = [
        "case",
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "candidate_delta_vs_baseline",
        "vanilla_delta_vs_baseline",
        "vanilla_minus_candidate_delta",
        "candidate_support_size",
        "vanilla_support_size",
        "support_intersection_size",
        "support_distance",
        "endpoint_distance",
        "boundary_anchor_rows",
        "boundary_anchor_truncated",
    ]
    lines.extend(_markdown_table(pair_rows[[c for c in pair_cols if c in pair_rows.columns]]))
    lines.extend(["", "## Support Class Counts", ""])
    if not node_rows.empty:
        support_counts = (
            node_rows.groupby(["support_class", "boundary_role"], as_index=False)
            .agg(nodes=("node", "size"), node_weight=("node_weight", "sum"))
            .sort_values(["support_class", "boundary_role"])
        )
        lines.extend(_markdown_table(support_counts))
    lines.extend(["", "## Vanilla-Extra Roles By Pair", ""])
    if not node_rows.empty:
        vanilla_extra_counts = (
            node_rows[node_rows["support_class"].eq("vanilla_extra")]
            .groupby(
                [
                    "case",
                    "candidate_index",
                    "vanilla_seed",
                    "vanilla_randomness",
                    "boundary_role",
                ],
                as_index=False,
            )
            .agg(nodes=("node", "size"), node_weight=("node_weight", "sum"))
            .sort_values(
                [
                    "candidate_index",
                    "vanilla_seed",
                    "vanilla_randomness",
                    "boundary_role",
                ]
            )
        )
        display_cols = [
            "candidate_index",
            "vanilla_seed",
            "vanilla_randomness",
            "boundary_role",
            "nodes",
            "node_weight",
        ]
        lines.extend(
            _markdown_table(
                vanilla_extra_counts[
                    [c for c in display_cols if c in vanilla_extra_counts.columns]
                ],
                max_rows=30,
            )
        )
    lines.extend(["", "## Top Bridge-Like Groups", ""])
    if not group_rows.empty:
        bridge_cols = [
            "case",
            "candidate_index",
            "vanilla_seed",
            "support_class",
            "boundary_role",
            "node_count",
            "node_weight_sum",
            "bridge_score_mean",
            "collateral_score_mean",
            "necessity_score_mean",
        ]
        bridge = group_rows[
            group_rows["boundary_role"].eq("bridge_like")
        ].sort_values(["bridge_score_mean", "node_count"], ascending=[False, False])
        lines.extend(_markdown_table(bridge[[c for c in bridge_cols if c in bridge.columns]], max_rows=15))
    lines.extend(["", "## Top Collateral-Like Groups", ""])
    if not group_rows.empty:
        collateral_cols = [
            "case",
            "candidate_index",
            "vanilla_seed",
            "support_class",
            "boundary_role",
            "node_count",
            "node_weight_sum",
            "bridge_score_mean",
            "collateral_score_mean",
            "necessity_score_mean",
        ]
        collateral = group_rows[
            group_rows["boundary_role"].eq("collateral_like")
        ].sort_values(
            ["collateral_score_mean", "node_count"],
            ascending=[False, False],
        )
        lines.extend(_markdown_table(collateral[[c for c in collateral_cols if c in collateral.columns]], max_rows=15))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- These rows are screening features, not acceptance criteria.",
            "- Shrink/expand operators should stay blocked unless bridge-like and collateral-like groups are separable and auditable.",
            "- Endpoint-near rows still require support and cost checks before any Dongdaemun-refinement claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _parse_candidate_indices(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument(
        "--candidate-indices",
        default="",
        help="Optional comma-separated candidate indices to include.",
    )
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--max-boundary-anchors", type=int, default=8192)
    parser.add_argument("--role-margin", type=float, default=0.05)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        landscape_dir=args.landscape_dir,
        vanilla_dir=args.vanilla_dir,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        candidate_indices=_parse_candidate_indices(args.candidate_indices),
        baseline_iterations=args.baseline_iterations,
        polish_iterations=args.polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        max_boundary_anchors=args.max_boundary_anchors,
        role_margin=args.role_margin,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
