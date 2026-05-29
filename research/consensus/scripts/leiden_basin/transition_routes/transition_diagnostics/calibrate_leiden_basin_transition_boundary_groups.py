#!/usr/bin/env python3
"""Calibrate boundary proxy scores with group-level dry-runs.

This is diagnostic-only. It takes boundary groups from
``analyze_leiden_basin_transition_boundaries.py`` and tests whether selected
vanilla-extra groups behave like removable collateral or necessary bridge
movement under small membership edits and short fixed-outside polish.
"""

from __future__ import annotations

import argparse
import json
import math
import time
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

from analyze_leiden_basin_transition_boundaries import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_BOUNDARY_DIR,
    GROUP_ROWS_FILENAME,
    NODE_ROWS_FILENAME,
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _best_partner_maps,
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
    _parse_vanilla_config,
    _recreate_candidate,
    _run_leiden,
    _safe_int,
    changed_support_nodes,
    endpoint_distance,
    fixed_outside,
    support_distance,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "basin_transition_boundary_calibration_field34_cc"
)
ROWS_FILENAME = "basin_transition_boundary_calibration_rows.csv"
SELECTED_GROUPS_FILENAME = "basin_transition_boundary_calibration_selected_groups.csv"
SUMMARY_FILENAME = "basin_transition_boundary_calibration_summary.json"
REPORT_FILENAME = "basin_transition_boundary_calibration_report.md"

GROUP_KEY_COLUMNS = [
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

ACTION_NAMES = (
    "vanilla_revert_baseline_aligned",
    "vanilla_revert_candidate_aligned",
    "candidate_add_vanilla_aligned",
)

def aligned_group_transplant(
    base_membership: np.ndarray,
    donor_membership: np.ndarray,
    group_nodes: np.ndarray | list[int],
) -> tuple[np.ndarray, int]:
    """Assign group nodes to donor-equivalent labels in the base label space."""
    base = np.asarray(base_membership, dtype=np.uint64)
    donor = np.asarray(donor_membership, dtype=np.uint64)
    nodes = np.asarray(group_nodes, dtype=np.int64)
    if base.shape != donor.shape:
        raise ValueError("base and donor memberships must have the same shape")
    out = base.copy()
    donor_to_base, _base_to_donor = _best_partner_maps(donor, base)
    next_label = int(out.max(initial=0)) + 1
    fresh_labels: dict[int, int] = {}
    fallback_count = 0
    for node in nodes:
        donor_label = int(donor[int(node)])
        target = donor_to_base.get(donor_label)
        if target is None:
            target = fresh_labels.get(donor_label)
            if target is None:
                target = next_label
                fresh_labels[donor_label] = target
                next_label += 1
            fallback_count += 1
        out[int(node)] = np.uint64(target)
    return out, fallback_count

def select_calibration_groups(
    group_rows: pd.DataFrame,
    *,
    roles: tuple[str, ...],
    max_groups_per_role: int,
    max_groups_total: int,
    min_node_count: int,
) -> pd.DataFrame:
    if group_rows.empty:
        return pd.DataFrame()
    frame = group_rows[group_rows["boundary_role"].isin(roles)].copy()
    frame = frame[
        pd.to_numeric(frame["node_count"], errors="coerce") >= min_node_count
    ]
    if frame.empty:
        return frame
    role_score_columns = {
        "bridge_like": "bridge_score_mean",
        "collateral_like": "collateral_score_mean",
        "ambiguous": "node_count",
    }
    selected: list[pd.DataFrame] = []
    pair_cols = [
        "case",
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "vanilla_requested_n_iterations",
    ]
    for (_pair, pair_frame) in frame.groupby(pair_cols, dropna=False):
        for role in roles:
            role_frame = pair_frame[pair_frame["boundary_role"].eq(role)].copy()
            if role_frame.empty:
                continue
            score_col = role_score_columns.get(role, "node_count")
            role_frame["_selection_score"] = pd.to_numeric(
                role_frame.get(score_col, 0.0),
                errors="coerce",
            ).fillna(0.0)
            role_frame = role_frame.sort_values(
                ["_selection_score", "node_count", "node_weight_sum"],
                ascending=[False, False, False],
            ).head(int(max_groups_per_role))
            selected.append(role_frame)
    if not selected:
        return pd.DataFrame()
    out = pd.concat(selected, ignore_index=True)
    out["selection_kind"] = "label_group"
    out["node_ids"] = ""
    out = out.sort_values(
        [
            "boundary_role",
            "_selection_score",
            "node_count",
            "node_weight_sum",
        ],
        ascending=[True, False, False, False],
    ).head(int(max_groups_total))
    out = out.reset_index(drop=True)
    out["calibration_group_id"] = [f"group_{idx:04d}" for idx in range(len(out))]
    return out

def select_role_chunks(
    node_rows: pd.DataFrame,
    *,
    roles: tuple[str, ...],
    chunk_sizes: tuple[int, ...],
    max_chunk_groups: int,
) -> pd.DataFrame:
    if node_rows.empty or not chunk_sizes or max_chunk_groups <= 0:
        return pd.DataFrame()
    score_columns = {
        "bridge_like": "bridge_score",
        "collateral_like": "collateral_score",
        "ambiguous": "necessity_score",
    }
    pair_cols = [
        "case",
        "field",
        "method",
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "vanilla_requested_n_iterations",
    ]
    rows: list[dict[str, Any]] = []
    for (_pair, pair_frame) in node_rows.groupby(pair_cols, dropna=False):
        for role in roles:
            role_frame = pair_frame[pair_frame["boundary_role"].eq(role)].copy()
            if role_frame.empty:
                continue
            score_col = score_columns.get(role, "incident_weight_total")
            role_frame["_selection_score"] = pd.to_numeric(
                role_frame.get(score_col, 0.0),
                errors="coerce",
            ).fillna(0.0)
            role_frame = role_frame.sort_values(
                ["_selection_score", "node_weight", "incident_weight_total"],
                ascending=[False, False, False],
            )
            for chunk_size in chunk_sizes:
                chunk = role_frame.head(int(chunk_size))
                if chunk.empty:
                    continue
                first = chunk.iloc[0]
                rows.append(
                    {
                        **{column: first[column] for column in pair_cols},
                        "support_class": "vanilla_extra",
                        "boundary_role": role,
                        "baseline_label": -1,
                        "candidate_label": -1,
                        "vanilla_label": -1,
                        "node_count": int(len(chunk)),
                        "node_weight_sum": float(chunk["node_weight"].sum()),
                        "incident_weight_total": float(
                            chunk["incident_weight_total"].sum()
                        ),
                        "bridge_score_mean": float(chunk["bridge_score"].mean()),
                        "collateral_score_mean": float(
                            chunk["collateral_score"].mean()
                        ),
                        "necessity_score_mean": float(chunk["necessity_score"].mean()),
                        "boundary_role_margin_mean": float(
                            chunk["boundary_role_margin"].mean()
                        ),
                        "_selection_score": float(chunk["_selection_score"].mean()),
                        "selection_kind": f"role_chunk_{int(chunk_size)}",
                        "node_ids": ",".join(str(int(node)) for node in chunk["node"]),
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(
        [
            "boundary_role",
            "_selection_score",
            "node_count",
            "node_weight_sum",
        ],
        ascending=[True, False, False, False],
    ).head(int(max_chunk_groups))
    out = out.reset_index(drop=True)
    return out

def combine_selected_groups(
    label_groups: pd.DataFrame,
    chunk_groups: pd.DataFrame,
) -> pd.DataFrame:
    frames = [frame for frame in (label_groups, chunk_groups) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["calibration_group_id"] = [f"group_{idx:04d}" for idx in range(len(out))]
    return out

def _nodes_for_group(node_rows: pd.DataFrame, group: pd.Series) -> np.ndarray:
    node_ids = str(group.get("node_ids", ""))
    if node_ids and node_ids.lower() != "nan":
        return np.asarray(
            [int(part) for part in node_ids.split(",") if part.strip()],
            dtype=np.uint32,
        )
    mask = np.ones(len(node_rows), dtype=np.bool_)
    for column in GROUP_KEY_COLUMNS:
        left = node_rows[column]
        right = group[column]
        if pd.isna(right):
            mask &= left.isna().to_numpy()
        else:
            mask &= left.eq(right).to_numpy()
    return np.asarray(node_rows.loc[mask, "node"], dtype=np.uint32)

def _action_initial_membership(
    *,
    action: str,
    group_nodes: np.ndarray,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
) -> tuple[np.ndarray, str, str, int]:
    if action == "vanilla_revert_baseline_aligned":
        initial, fallback = aligned_group_transplant(
            vanilla_membership,
            baseline_membership,
            group_nodes,
        )
        return initial, "vanilla", "baseline", fallback
    if action == "vanilla_revert_candidate_aligned":
        initial, fallback = aligned_group_transplant(
            vanilla_membership,
            candidate_membership,
            group_nodes,
        )
        return initial, "vanilla", "candidate", fallback
    if action == "candidate_add_vanilla_aligned":
        initial, fallback = aligned_group_transplant(
            candidate_membership,
            vanilla_membership,
            group_nodes,
        )
        return initial, "candidate", "vanilla", fallback
    raise ValueError(f"Unsupported calibration action: {action}")

def _quality_for_base(
    *,
    base_kind: str,
    candidate_quality: float,
    vanilla_quality: float,
) -> float:
    if base_kind == "candidate":
        return float(candidate_quality)
    if base_kind == "vanilla":
        return float(vanilla_quality)
    raise ValueError(f"Unsupported base kind: {base_kind}")

def _run_calibration_action(
    *,
    graph: Any,
    action: str,
    group_nodes: np.ndarray,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    baseline_quality: float,
    candidate_quality: float,
    vanilla_quality: float,
    sketch_nodes: np.ndarray,
    resolution: float,
    seed: int,
    iterations: int,
    randomness: float,
) -> dict[str, Any]:
    initial, base_kind, donor_kind, fallback_count = _action_initial_membership(
        action=action,
        group_nodes=group_nodes,
        baseline_membership=baseline_membership,
        candidate_membership=candidate_membership,
        vanilla_membership=vanilla_membership,
    )
    base_quality = _quality_for_base(
        base_kind=base_kind,
        candidate_quality=candidate_quality,
        vanilla_quality=vanilla_quality,
    )
    initial_quality = float(graph.cpm_quality(initial, resolution=float(resolution)))
    if iterations <= 0:
        result_membership = initial
        result_quality = initial_quality
        elapsed_sec = 0.0
    else:
        start = time.perf_counter()
        result = graph.run_leiden(
            resolution=float(resolution),
            seed=int(seed),
            n_iterations=int(iterations),
            randomness=float(randomness),
            initial_membership=initial,
            fixed_nodes=fixed_outside(len(initial), group_nodes),
        )
        elapsed_sec = float(time.perf_counter() - start)
        result_membership = np.asarray(result.membership, dtype=np.uint64)
        result_quality = float(result.quality)
    candidate_support = changed_support_nodes(baseline_membership, candidate_membership)
    vanilla_support = changed_support_nodes(baseline_membership, vanilla_membership)
    result_support = changed_support_nodes(baseline_membership, result_membership)
    dist_candidate, inter_candidate, union_candidate = support_distance(
        result_support,
        candidate_support,
    )
    dist_vanilla, inter_vanilla, union_vanilla = support_distance(
        result_support,
        vanilla_support,
    )
    return {
        "action": action,
        "base_kind": base_kind,
        "donor_kind": donor_kind,
        "label_fallback_count": int(fallback_count),
        "base_quality": base_quality,
        "initial_quality": initial_quality,
        "quality": result_quality,
        "delta_initial_vs_base": initial_quality - base_quality,
        "delta_vs_base": result_quality - base_quality,
        "delta_vs_baseline": result_quality - baseline_quality,
        "delta_vs_candidate": result_quality - candidate_quality,
        "delta_vs_vanilla": result_quality - vanilla_quality,
        "elapsed_sec": elapsed_sec,
        "result_support_size": int(result_support.size),
        "result_support_distance_to_candidate": float(dist_candidate),
        "result_support_intersection_with_candidate": int(inter_candidate),
        "result_support_union_with_candidate": int(union_candidate),
        "result_support_distance_to_vanilla": float(dist_vanilla),
        "result_support_intersection_with_vanilla": int(inter_vanilla),
        "result_support_union_with_vanilla": int(union_vanilla),
        "result_endpoint_distance_to_candidate": endpoint_distance(
            result_membership,
            candidate_membership,
            sketch_nodes,
        ),
        "result_endpoint_distance_to_vanilla": endpoint_distance(
            result_membership,
            vanilla_membership,
            sketch_nodes,
        ),
    }

def _group_metadata(group: pd.Series, group_nodes: np.ndarray, rank: int) -> dict[str, Any]:
    return {
        "calibration_group_id": group["calibration_group_id"],
        "selection_rank": int(rank),
        "selection_kind": group.get("selection_kind", "label_group"),
        "case": group["case"],
        "field": group["field"],
        "method": group["method"],
        "candidate_index": int(group["candidate_index"]),
        "vanilla_seed": int(group["vanilla_seed"]),
        "vanilla_randomness": float(group["vanilla_randomness"]),
        "vanilla_requested_n_iterations": group["vanilla_requested_n_iterations"],
        "support_class": group["support_class"],
        "boundary_role": group["boundary_role"],
        "baseline_label": int(group["baseline_label"]),
        "candidate_label": int(group["candidate_label"]),
        "vanilla_label": int(group["vanilla_label"]),
        "group_node_count": int(len(group_nodes)),
        "group_node_weight": float(group.get("node_weight_sum", len(group_nodes))),
        "bridge_score_mean": float(group.get("bridge_score_mean", math.nan)),
        "collateral_score_mean": float(group.get("collateral_score_mean", math.nan)),
        "necessity_score_mean": float(group.get("necessity_score_mean", math.nan)),
        "boundary_role_margin_mean": float(
            group.get("boundary_role_margin_mean", math.nan)
        ),
    }

def _pair_lookup_key(row: pd.Series) -> tuple[str, int, int, float, str]:
    return (
        str(row["case"]),
        int(row["candidate_index"]),
        int(row["vanilla_seed"]),
        float(row["vanilla_randomness"]),
        str(row["vanilla_requested_n_iterations"]),
    )

def _hypothesis_by_pair(landscape_dir: Path) -> dict[tuple[str, int, int, float, str], pd.Series]:
    hypotheses = pd.read_csv(landscape_dir / HYPOTHESES_FILENAME)
    out: dict[tuple[str, int, int, float, str], pd.Series] = {}
    for _, row in hypotheses.iterrows():
        if row.get("hypothesis") != "candidate_local_core_inside_broader_vanilla":
            continue
        candidate_index = int(str(row["candidate_node_id"]).rsplit(":", 1)[-1])
        seed, randomness, n_iterations = _parse_vanilla_config(row["vanilla_node_id"])
        out[(str(row["case"]), candidate_index, seed, randomness, str(n_iterations))] = row
    return out

def run_calibration(
    *,
    candidate_dirs: tuple[Path, ...],
    boundary_dir: Path,
    landscape_dir: Path,
    vanilla_dir: Path,
    output_dir: Path,
    roles: tuple[str, ...],
    actions: tuple[str, ...],
    max_groups_per_role: int,
    max_groups_total: int,
    min_node_count: int,
    include_role_chunks: bool,
    chunk_sizes: tuple[int, ...],
    max_chunk_groups: int,
    baseline_iterations: int,
    polish_iterations: int,
    calibration_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    calibration_seed_offset: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    node_rows = pd.read_csv(boundary_dir / NODE_ROWS_FILENAME)
    group_rows = pd.read_csv(boundary_dir / GROUP_ROWS_FILENAME)
    label_groups = select_calibration_groups(
        group_rows,
        roles=roles,
        max_groups_per_role=max_groups_per_role,
        max_groups_total=max_groups_total,
        min_node_count=min_node_count,
    )
    chunk_groups = (
        select_role_chunks(
            node_rows,
            roles=roles,
            chunk_sizes=chunk_sizes,
            max_chunk_groups=max_chunk_groups,
        )
        if include_role_chunks
        else pd.DataFrame()
    )
    selected_groups = combine_selected_groups(label_groups, chunk_groups)
    if selected_groups.empty:
        raise ValueError("No boundary groups selected for calibration")
    selected_groups.to_csv(output_dir / SELECTED_GROUPS_FILENAME, index=False)

    candidates = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)
    hypotheses = _hypothesis_by_pair(landscape_dir)
    baseline_cache: dict[str, Any] = {}
    candidate_cache: dict[tuple[str, int], Any] = {}
    vanilla_cache: dict[tuple[str, int, float, str], Any] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, Any]] = {}
    out_rows: list[dict[str, Any]] = []

    for rank, group in selected_groups.iterrows():
        group_nodes = _nodes_for_group(node_rows, group)
        if group_nodes.size == 0:
            continue
        key = _pair_lookup_key(group)
        hypothesis = hypotheses.get(key)
        if hypothesis is None:
            raise ValueError(f"Missing transition hypothesis for {key}")
        case, candidate_index, seed, vanilla_randomness, vanilla_n = key
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

        metadata = _group_metadata(group, group_nodes, int(rank))
        for action_index, action in enumerate(actions):
            calibration_seed = (
                int(candidate_row.get("seed", 0))
                + int(calibration_seed_offset)
                + int(candidate_index) * 101
                + int(rank) * 17
                + action_index
            )
            action_row = _run_calibration_action(
                graph=graph,
                action=action,
                group_nodes=group_nodes,
                baseline_membership=baseline.membership,
                candidate_membership=candidate.recreated.membership,
                vanilla_membership=vanilla.membership,
                baseline_quality=baseline.quality,
                candidate_quality=candidate.recreated.quality,
                vanilla_quality=vanilla.quality,
                sketch_nodes=sketch_nodes,
                resolution=resolution,
                seed=calibration_seed,
                iterations=calibration_iterations,
                randomness=randomness,
            )
            out_rows.append(
                {
                    **metadata,
                    **action_row,
                    "calibration_seed": int(calibration_seed),
                    "calibration_iterations": int(calibration_iterations),
                }
            )

    rows = pd.DataFrame(out_rows)
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    summary = {
        "schema": "leiden_basin_transition_boundary_calibration.v1",
        "boundary_dir": str(boundary_dir),
        "landscape_dir": str(landscape_dir),
        "vanilla_dir": str(vanilla_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "selected_group_rows": int(len(selected_groups)),
        "calibration_rows": int(len(rows)),
        "roles": list(roles),
        "actions": list(actions),
        "max_groups_per_role": int(max_groups_per_role),
        "max_groups_total": int(max_groups_total),
        "min_node_count": int(min_node_count),
        "include_role_chunks": bool(include_role_chunks),
        "chunk_sizes": [int(value) for value in chunk_sizes],
        "max_chunk_groups": int(max_chunk_groups),
        "baseline_iterations": int(baseline_iterations),
        "polish_iterations": int(polish_iterations),
        "calibration_iterations": int(calibration_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "output_dir": str(output_dir),
    }
    if not rows.empty:
        summary["selected_groups_by_kind"] = {
            str(key): int(value)
            for key, value in selected_groups["selection_kind"].value_counts().items()
        }
        summary["rows_by_action"] = {
            str(key): int(value) for key, value in rows["action"].value_counts().items()
        }
        summary["delta_vs_base_by_action_median"] = {
            str(key): float(value)
            for key, value in rows.groupby("action")["delta_vs_base"].median().items()
        }
        summary["positive_delta_vs_base_rows"] = int(
            (rows["delta_vs_base"] > 1e-12).sum()
        )
        summary["max_delta_vs_base"] = float(rows["delta_vs_base"].max())
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / REPORT_FILENAME, rows, selected_groups)
    return summary

def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 25) -> list[str]:
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
    rows: pd.DataFrame,
    selected_groups: pd.DataFrame,
) -> None:
    lines = [
        "# Basin Transition Boundary Calibration",
        "",
        "This is a diagnostic dry run. It tests selected boundary groups with small aligned membership edits and fixed-outside polish.",
        "",
        "## Selected Groups",
        "",
    ]
    selected_cols = [
        "calibration_group_id",
        "selection_kind",
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "boundary_role",
        "node_count",
        "node_weight_sum",
        "bridge_score_mean",
        "collateral_score_mean",
        "necessity_score_mean",
    ]
    lines.extend(
        _markdown_table(
            selected_groups[[c for c in selected_cols if c in selected_groups.columns]]
        )
    )
    lines.extend(["", "## Action Summary", ""])
    if not rows.empty:
        summary = (
            rows.groupby(["action", "boundary_role"], as_index=False)
            .agg(
                rows=("action", "size"),
                delta_initial_vs_base_median=("delta_initial_vs_base", "median"),
                delta_vs_base_median=("delta_vs_base", "median"),
                delta_vs_base_min=("delta_vs_base", "min"),
                delta_vs_base_max=("delta_vs_base", "max"),
                positive_delta_vs_base_rows=(
                    "delta_vs_base",
                    lambda values: int((values > 1e-12).sum()),
                ),
                delta_vs_candidate_median=("delta_vs_candidate", "median"),
                delta_vs_vanilla_median=("delta_vs_vanilla", "median"),
                support_distance_to_candidate_median=(
                    "result_support_distance_to_candidate",
                    "median",
                ),
                support_distance_to_vanilla_median=(
                    "result_support_distance_to_vanilla",
                    "median",
                ),
                elapsed_sec_median=("elapsed_sec", "median"),
            )
            .sort_values(["action", "boundary_role"])
        )
        lines.extend(_markdown_table(summary))
    lines.extend(["", "## Rows", ""])
    if not rows.empty:
        display_cols = [
            "calibration_group_id",
            "selection_kind",
            "boundary_role",
            "group_node_count",
            "action",
            "delta_initial_vs_base",
            "delta_vs_base",
            "delta_vs_candidate",
            "delta_vs_vanilla",
            "result_support_distance_to_candidate",
            "result_support_distance_to_vanilla",
            "elapsed_sec",
        ]
        display = rows.sort_values(
            ["boundary_role", "calibration_group_id", "action"]
        )
        lines.extend(
            _markdown_table(display[[c for c in display_cols if c in display.columns]])
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- These rows calibrate proxy scores; they are not accepted outputs.",
            "- A collateral-like proxy should not be treated as removable unless revert/add dry-runs preserve quality and improve the cost/support tradeoff.",
            "- A shrink/expand operator still needs seed-control comparison before any Dongdaemun-refinement claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _parse_csv_tuple(value: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())

def _parse_int_tuple(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value.strip():
        return default
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--roles",
        default="bridge_like,ambiguous,collateral_like",
        help="Comma-separated boundary roles to calibrate.",
    )
    parser.add_argument(
        "--actions",
        default=",".join(ACTION_NAMES),
        help="Comma-separated calibration actions.",
    )
    parser.add_argument("--max-groups-per-role", type=int, default=2)
    parser.add_argument("--max-groups-total", type=int, default=12)
    parser.add_argument("--min-node-count", type=int, default=1)
    parser.add_argument(
        "--include-role-chunks",
        action="store_true",
        help="Also calibrate top role-level chunks per pair.",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="16",
        help="Comma-separated role chunk sizes when --include-role-chunks is set.",
    )
    parser.add_argument("--max-chunk-groups", type=int, default=12)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--calibration-iterations", type=int, default=2)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--calibration-seed-offset", type=int, default=7000)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_calibration(
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        boundary_dir=args.boundary_dir,
        landscape_dir=args.landscape_dir,
        vanilla_dir=args.vanilla_dir,
        output_dir=args.output_dir,
        roles=_parse_csv_tuple(
            args.roles,
            ("bridge_like", "ambiguous", "collateral_like"),
        ),
        actions=_parse_csv_tuple(args.actions, ACTION_NAMES),
        max_groups_per_role=args.max_groups_per_role,
        max_groups_total=args.max_groups_total,
        min_node_count=args.min_node_count,
        include_role_chunks=bool(args.include_role_chunks),
        chunk_sizes=_parse_int_tuple(args.chunk_sizes, (16,)),
        max_chunk_groups=args.max_chunk_groups,
        baseline_iterations=args.baseline_iterations,
        polish_iterations=args.polish_iterations,
        calibration_iterations=args.calibration_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        calibration_seed_offset=args.calibration_seed_offset,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
