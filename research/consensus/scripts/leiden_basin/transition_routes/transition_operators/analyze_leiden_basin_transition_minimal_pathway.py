#!/usr/bin/env python3
"""Compute diagnostic minimum support-edit pathways between basin endpoints.

This does not ask whether Leiden can naturally perform the transition. It uses
collision-safe fresh-label edits for `S_V - S_C` units, then measures what exact
CPM quality barrier appears along that relabel pathway.
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

from analyze_leiden_basin_transition_boundaries import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_BOUNDARY_DIR,
    GROUP_ROWS_FILENAME,
    NODE_ROWS_FILENAME,
)
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
    "basin_transition_minimal_pathway_field34_cc"
)
STEP_ROWS_FILENAME = "basin_transition_minimal_pathway_steps.csv"
PAIR_ROWS_FILENAME = "basin_transition_minimal_pathway_pairs.csv"
UNIT_ROWS_FILENAME = "basin_transition_minimal_pathway_units.csv"
SUMMARY_FILENAME = "basin_transition_minimal_pathway_summary.json"
REPORT_FILENAME = "basin_transition_minimal_pathway_report.md"

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

ACTION_NAMES = ("baseline_forced", "candidate_forced")
ORDERING_POLICIES = ("least_direct_debt", "proxy_collateral_desc")

def _nodes_for_group(node_rows: pd.DataFrame, group: pd.Series) -> np.ndarray:
    mask = np.ones(len(node_rows), dtype=np.bool_)
    for column in GROUP_KEY_COLUMNS:
        left = node_rows[column]
        right = group[column]
        if pd.isna(right):
            mask &= left.isna().to_numpy()
        else:
            mask &= left.eq(right).to_numpy()
    return np.asarray(node_rows.loc[mask, "node"], dtype=np.uint32)

def pathway_units_for_pair(
    *,
    node_rows: pd.DataFrame,
    group_rows: pd.DataFrame,
    pair_key: dict[str, Any],
) -> pd.DataFrame:
    """Return vanilla-extra edit units needed to remove V-only support."""
    pair_mask = np.ones(len(group_rows), dtype=np.bool_)
    for column, value in pair_key.items():
        if column == "vanilla_requested_n_iterations":
            pair_mask &= group_rows[column].astype(str).eq(str(value)).to_numpy()
        else:
            pair_mask &= group_rows[column].eq(value).to_numpy()
    frame = group_rows[
        pair_mask
        & group_rows["support_class"].eq("vanilla_extra").to_numpy()
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for unit_index, (_, group) in enumerate(frame.iterrows()):
        nodes = _nodes_for_group(node_rows, group)
        if nodes.size == 0:
            continue
        rows.append(
            {
                **{column: group[column] for column in GROUP_KEY_COLUMNS},
                "unit_id": f"unit_{unit_index:05d}",
                "unit_node_count": int(nodes.size),
                "unit_node_weight": float(group.get("node_weight_sum", nodes.size)),
                "bridge_score_mean": float(group.get("bridge_score_mean", math.nan)),
                "collateral_score_mean": float(
                    group.get("collateral_score_mean", math.nan)
                ),
                "necessity_score_mean": float(
                    group.get("necessity_score_mean", math.nan)
                ),
                "boundary_role_margin_mean": float(
                    group.get("boundary_role_margin_mean", math.nan)
                ),
                "node_ids": ",".join(str(int(node)) for node in nodes),
            }
        )
    return pd.DataFrame(rows)

def _parse_node_ids(value: Any) -> np.ndarray:
    text = str(value)
    if not text or text.lower() == "nan":
        return np.asarray([], dtype=np.uint32)
    return np.asarray([int(part) for part in text.split(",") if part], dtype=np.uint32)

def _fresh_group_transplant(
    membership: np.ndarray,
    donor_membership: np.ndarray,
    nodes: np.ndarray,
    *,
    label_map: dict[int, int] | None = None,
    next_label: int | None = None,
) -> tuple[np.ndarray, dict[int, int], int]:
    """Force donor labels into a fresh namespace for edited nodes.

    The label map is cumulative across pathway steps, so two units with the same
    donor label join the same fresh target label instead of being over-split.
    """
    out = np.asarray(membership, dtype=np.uint64).copy()
    donor = np.asarray(donor_membership, dtype=np.uint64)
    mapping: dict[int, int] = dict(label_map or {})
    next_out_label = (
        int(next_label)
        if next_label is not None
        else int(out.max(initial=0)) + 1
    )
    for node in np.asarray(nodes, dtype=np.int64):
        donor_label = int(donor[int(node)])
        target = mapping.get(donor_label)
        if target is None:
            target = next_out_label
            mapping[donor_label] = target
            next_out_label += 1
        out[int(node)] = np.uint64(target)
    return out, mapping, next_out_label

def _donor_for_action(
    *,
    action: str,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
) -> tuple[str, np.ndarray]:
    if action == "baseline_forced":
        return "baseline", baseline_membership
    if action == "candidate_forced":
        return "candidate", candidate_membership
    raise ValueError(f"Unsupported pathway action: {action}")

def score_unit_direct_delta(
    *,
    graph: Any,
    start_membership: np.ndarray,
    start_quality: float,
    unit: pd.Series,
    action: str,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    resolution: float,
) -> float:
    nodes = _parse_node_ids(unit["node_ids"])
    _donor_name, donor = _donor_for_action(
        action=action,
        baseline_membership=baseline_membership,
        candidate_membership=candidate_membership,
    )
    proposed, _mapping, _next_label = _fresh_group_transplant(
        start_membership,
        donor,
        nodes=nodes,
    )
    quality = float(graph.cpm_quality(proposed, resolution=float(resolution)))
    return quality - float(start_quality)

def order_units(
    units: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    if units.empty:
        return units.copy()
    frame = units.copy()
    if policy == "least_direct_debt":
        return frame.sort_values(
            [
                "direct_delta_q",
                "unit_node_count",
                "collateral_score_mean",
            ],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    if policy == "proxy_collateral_desc":
        return frame.sort_values(
            [
                "collateral_score_mean",
                "direct_delta_q",
                "unit_node_count",
            ],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    raise ValueError(f"Unsupported ordering policy: {policy}")

def _pathway_pair_context(row: pd.Series) -> dict[str, Any]:
    return {
        "case": row["case"],
        "field": row["field"],
        "method": row["method"],
        "candidate_index": int(row["candidate_index"]),
        "vanilla_seed": int(row["vanilla_seed"]),
        "vanilla_randomness": float(row["vanilla_randomness"]),
        "vanilla_requested_n_iterations": row["vanilla_requested_n_iterations"],
    }

def compute_cumulative_pathway(
    *,
    graph: Any,
    units: pd.DataFrame,
    action: str,
    ordering_policy: str,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    baseline_quality: float,
    candidate_quality: float,
    vanilla_quality: float,
    sketch_nodes: np.ndarray,
    resolution: float,
    context: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply all units cumulatively and record quality/support barrier."""
    ordered = order_units(units, policy=ordering_policy)
    current = np.asarray(vanilla_membership, dtype=np.uint64).copy()
    start_quality = float(vanilla_quality)
    min_quality = start_quality
    max_quality = start_quality
    _donor_name, donor_membership = _donor_for_action(
        action=action,
        baseline_membership=baseline_membership,
        candidate_membership=candidate_membership,
    )
    label_map: dict[int, int] = {}
    next_label = int(np.max(vanilla_membership, initial=0)) + 1
    candidate_support = changed_support_nodes(baseline_membership, candidate_membership)
    vanilla_support = changed_support_nodes(baseline_membership, vanilla_membership)
    total_nodes = (
        int(sum(int(v) for v in ordered["unit_node_count"]))
        if not ordered.empty
        else 0
    )
    step_rows: list[dict[str, Any]] = []
    edited_nodes = 0
    for step_index, (_, unit) in enumerate(ordered.iterrows(), start=1):
        nodes = _parse_node_ids(unit["node_ids"])
        current, label_map, next_label = _fresh_group_transplant(
            current,
            donor_membership,
            nodes=nodes,
            label_map=label_map,
            next_label=next_label,
        )
        edited_nodes += int(nodes.size)
        quality = float(graph.cpm_quality(current, resolution=float(resolution)))
        min_quality = min(min_quality, quality)
        max_quality = max(max_quality, quality)
        result_support = changed_support_nodes(baseline_membership, current)
        dist_candidate, inter_candidate, union_candidate = support_distance(
            result_support,
            candidate_support,
        )
        dist_vanilla, inter_vanilla, union_vanilla = support_distance(
            result_support,
            vanilla_support,
        )
        step_rows.append(
            {
                **context,
                "direction": "vanilla_to_candidate_support",
                "action": action,
                "ordering_policy": ordering_policy,
                "step_index": int(step_index),
                "unit_id": unit["unit_id"],
                "boundary_role": unit["boundary_role"],
                "unit_node_count": int(nodes.size),
                "edited_node_count": int(edited_nodes),
                "edited_node_fraction": (
                    float(edited_nodes) / float(total_nodes) if total_nodes else math.nan
                ),
                "direct_delta_q": float(unit["direct_delta_q"]),
                "quality": quality,
                "delta_vs_start": quality - start_quality,
                "delta_vs_candidate": quality - candidate_quality,
                "delta_vs_vanilla": quality - vanilla_quality,
                "quality_debt_vs_start": max(0.0, start_quality - quality),
                "min_quality_so_far": min_quality,
                "quality_barrier_so_far": max(0.0, start_quality - min_quality),
                "result_support_size": int(result_support.size),
                "result_support_distance_to_candidate": float(dist_candidate),
                "result_support_intersection_with_candidate": int(inter_candidate),
                "result_support_union_with_candidate": int(union_candidate),
                "result_support_distance_to_vanilla": float(dist_vanilla),
                "result_support_intersection_with_vanilla": int(inter_vanilla),
                "result_support_union_with_vanilla": int(union_vanilla),
                "endpoint_distance_to_candidate": endpoint_distance(
                    current,
                    candidate_membership,
                    sketch_nodes,
                ),
                "endpoint_distance_to_vanilla": endpoint_distance(
                    current,
                    vanilla_membership,
                    sketch_nodes,
                ),
            }
        )
    final_support = changed_support_nodes(baseline_membership, current)
    final_support_set = {int(node) for node in final_support}
    candidate_support_set = {int(node) for node in candidate_support}
    residual_extra_support = final_support_set - candidate_support_set
    missing_candidate_support = candidate_support_set - final_support_set
    final_dist_candidate, final_inter_candidate, final_union_candidate = support_distance(
        final_support,
        candidate_support,
    )
    final_dist_vanilla, _final_inter_vanilla, _final_union_vanilla = support_distance(
        final_support,
        vanilla_support,
    )
    final_quality = float(graph.cpm_quality(current, resolution=float(resolution)))
    pair_summary = {
        **context,
        "direction": "vanilla_to_candidate_support",
        "action": action,
        "ordering_policy": ordering_policy,
        "start_quality": start_quality,
        "candidate_quality": float(candidate_quality),
        "baseline_quality": float(baseline_quality),
        "total_units": int(len(ordered)),
        "node_edit_lower_bound": int(total_nodes),
        "residual_extra_support_after_path": int(len(residual_extra_support)),
        "missing_candidate_support_after_path": int(len(missing_candidate_support)),
        "closure_node_edit_lower_bound": int(
            total_nodes + len(residual_extra_support)
        ),
        "candidate_support_size": int(candidate_support.size),
        "vanilla_support_size": int(vanilla_support.size),
        "final_quality": final_quality,
        "final_delta_vs_start": float(final_quality - start_quality),
        "min_quality": float(min_quality),
        "max_quality": float(max_quality),
        "quality_barrier": float(max(0.0, start_quality - min_quality)),
        "quality_peak_gain": float(max_quality - start_quality),
        "final_support_size": int(final_support.size),
        "final_support_distance_to_candidate": float(final_dist_candidate),
        "final_support_intersection_with_candidate": int(final_inter_candidate),
        "final_support_union_with_candidate": int(final_union_candidate),
        "final_support_distance_to_vanilla": float(final_dist_vanilla),
    }
    steps = pd.DataFrame(step_rows)
    for threshold in (0.5, 0.1, 0.05, 0.01, 0.001, 0.0):
        column = f"first_step_support_distance_le_{threshold:g}"
        if steps.empty:
            pair_summary[column] = math.nan
            continue
        hits = steps[steps["result_support_distance_to_candidate"] <= threshold]
        pair_summary[column] = (
            int(hits["step_index"].iloc[0]) if not hits.empty else math.nan
        )
    return steps, pair_summary

def _hypothesis_rows(
    landscape_dir: Path,
    *,
    max_pairs: int,
    candidate_indices: set[int],
) -> pd.DataFrame:
    hypotheses = pd.read_csv(landscape_dir / HYPOTHESES_FILENAME)
    return _select_hypotheses(
        hypotheses,
        max_pairs=max_pairs,
        candidate_indices=candidate_indices,
    )

def _parse_candidate_indices(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}

def _parse_csv_tuple(value: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())

def run_analysis(
    *,
    candidate_dirs: tuple[Path, ...],
    boundary_dir: Path,
    landscape_dir: Path,
    vanilla_dir: Path,
    output_dir: Path,
    max_pairs: int,
    candidate_indices: set[int],
    actions: tuple[str, ...],
    ordering_policies: tuple[str, ...],
    baseline_iterations: int,
    polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    node_rows = pd.read_csv(boundary_dir / NODE_ROWS_FILENAME)
    group_rows = pd.read_csv(boundary_dir / GROUP_ROWS_FILENAME)
    hypotheses = _hypothesis_rows(
        landscape_dir,
        max_pairs=max_pairs,
        candidate_indices=candidate_indices,
    )
    if hypotheses.empty:
        raise ValueError("No transition hypotheses selected")
    candidates = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)

    baseline_cache: dict[str, Any] = {}
    candidate_cache: dict[tuple[str, int], Any] = {}
    vanilla_cache: dict[tuple[str, int, float, str], Any] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, Any]] = {}
    all_steps: list[pd.DataFrame] = []
    all_units: list[pd.DataFrame] = []
    pair_summaries: list[dict[str, Any]] = []

    for _, hypothesis in hypotheses.iterrows():
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

        pair_key = {
            "case": case,
            "candidate_index": int(candidate_index),
            "vanilla_seed": int(seed),
            "vanilla_randomness": float(vanilla_randomness),
            "vanilla_requested_n_iterations": vanilla_n,
        }
        units = pathway_units_for_pair(
            node_rows=node_rows,
            group_rows=group_rows,
            pair_key=pair_key,
        )
        if units.empty:
            continue
        context = {
            "case": case,
            "field": hypothesis.get("field"),
            "method": hypothesis.get("method"),
            "candidate_index": int(candidate_index),
            "vanilla_seed": int(seed),
            "vanilla_randomness": float(vanilla_randomness),
            "vanilla_requested_n_iterations": vanilla_n,
        }
        for action in actions:
            scored_units = units.copy()
            scored_units["action"] = action
            scored_units["direct_delta_q"] = [
                score_unit_direct_delta(
                    graph=graph,
                    start_membership=vanilla.membership,
                    start_quality=vanilla.quality,
                    unit=unit,
                    action=action,
                    baseline_membership=baseline.membership,
                    candidate_membership=candidate.recreated.membership,
                    resolution=resolution,
                )
                for _, unit in scored_units.iterrows()
            ]
            all_units.append(scored_units)
            for policy in ordering_policies:
                steps, summary = compute_cumulative_pathway(
                    graph=graph,
                    units=scored_units,
                    action=action,
                    ordering_policy=policy,
                    baseline_membership=baseline.membership,
                    candidate_membership=candidate.recreated.membership,
                    vanilla_membership=vanilla.membership,
                    baseline_quality=baseline.quality,
                    candidate_quality=candidate.recreated.quality,
                    vanilla_quality=vanilla.quality,
                    sketch_nodes=sketch_nodes,
                    resolution=resolution,
                    context=context,
                )
                if not steps.empty:
                    all_steps.append(steps)
                pair_summaries.append(summary)

    step_rows = (
        pd.concat(all_steps, ignore_index=True)
        if all_steps
        else pd.DataFrame()
    )
    unit_rows = (
        pd.concat(all_units, ignore_index=True)
        if all_units
        else pd.DataFrame()
    )
    pair_rows = pd.DataFrame(pair_summaries)
    step_rows.to_csv(output_dir / STEP_ROWS_FILENAME, index=False)
    unit_rows.to_csv(output_dir / UNIT_ROWS_FILENAME, index=False)
    pair_rows.to_csv(output_dir / PAIR_ROWS_FILENAME, index=False)
    summary = {
        "schema": "leiden_basin_transition_minimal_pathway.v1",
        "boundary_dir": str(boundary_dir),
        "landscape_dir": str(landscape_dir),
        "vanilla_dir": str(vanilla_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "pair_rows": int(len(pair_rows)),
        "step_rows": int(len(step_rows)),
        "unit_rows": int(len(unit_rows)),
        "actions": list(actions),
        "ordering_policies": list(ordering_policies),
        "baseline_iterations": int(baseline_iterations),
        "polish_iterations": int(polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "output_dir": str(output_dir),
    }
    if not pair_rows.empty:
        summary["median_node_edit_lower_bound"] = float(
            pair_rows["node_edit_lower_bound"].median()
        )
        summary["median_residual_extra_support_after_path"] = float(
            pair_rows["residual_extra_support_after_path"].median()
        )
        summary["median_closure_node_edit_lower_bound"] = float(
            pair_rows["closure_node_edit_lower_bound"].median()
        )
        summary["median_quality_barrier"] = float(pair_rows["quality_barrier"].median())
        summary["min_final_support_distance_to_candidate"] = float(
            pair_rows["final_support_distance_to_candidate"].min()
        )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / REPORT_FILENAME, pair_rows, step_rows, unit_rows)
    return summary

def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 24) -> list[str]:
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
    step_rows: pd.DataFrame,
    unit_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Basin Transition Minimal Pathway",
        "",
        "This diagnostic computes collision-safe relabel pathways over direct `S_V - S_C` support units. It does not test whether Leiden naturally takes the pathway.",
        "",
        "## Pair Pathway Summary",
        "",
    ]
    pair_cols = [
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "action",
        "ordering_policy",
        "node_edit_lower_bound",
        "residual_extra_support_after_path",
        "missing_candidate_support_after_path",
        "closure_node_edit_lower_bound",
        "total_units",
        "quality_barrier",
        "quality_peak_gain",
        "final_delta_vs_start",
        "candidate_support_size",
        "vanilla_support_size",
        "final_support_distance_to_candidate",
        "final_support_distance_to_vanilla",
        "first_step_support_distance_le_0.1",
        "first_step_support_distance_le_0.01",
        "first_step_support_distance_le_0",
    ]
    display = (
        pair_rows.sort_values(
            [
                "quality_barrier",
                "final_support_distance_to_candidate",
                "node_edit_lower_bound",
            ],
            ascending=[True, True, True],
        )
        if not pair_rows.empty
        else pair_rows
    )
    lines.extend(_markdown_table(display[[c for c in pair_cols if c in display.columns]]))
    lines.extend(["", "## Unit Direct Delta Summary", ""])
    if not unit_rows.empty:
        unit_summary = (
            unit_rows.groupby(["action", "boundary_role"], as_index=False)
            .agg(
                units=("unit_id", "size"),
                nodes=("unit_node_count", "sum"),
                direct_delta_q_median=("direct_delta_q", "median"),
                direct_delta_q_min=("direct_delta_q", "min"),
                direct_delta_q_max=("direct_delta_q", "max"),
            )
            .sort_values(["action", "boundary_role"])
        )
        lines.extend(_markdown_table(unit_summary))
    lines.extend(["", "## Largest Barriers", ""])
    if not pair_rows.empty:
        barrier = pair_rows.sort_values(["quality_barrier"], ascending=False)
        lines.extend(_markdown_table(barrier[[c for c in pair_cols if c in barrier.columns]]))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- `node_edit_lower_bound` is exact for the selected support-unit target `S_V - S_C`.",
            "- `closure_node_edit_lower_bound` is a residual count under the fresh-label pathway, not the pure support-set lower bound.",
            "- `quality_barrier` is measured along deterministic greedy orderings, not a proof of the global optimal barrier.",
            "- Use the closure-context artifact for support-set lower bounds and label-context scale.",
            "- This artifact should guide the next mutable-set design; it is not an accepted Dongdaemun output.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument(
        "--candidate-indices",
        default="",
        help="Optional comma-separated candidate indices to include.",
    )
    parser.add_argument(
        "--actions",
        default=",".join(ACTION_NAMES),
        help="Comma-separated pathway edit actions.",
    )
    parser.add_argument(
        "--ordering-policies",
        default=",".join(ORDERING_POLICIES),
        help="Comma-separated ordering policies.",
    )
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        boundary_dir=args.boundary_dir,
        landscape_dir=args.landscape_dir,
        vanilla_dir=args.vanilla_dir,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        candidate_indices=_parse_candidate_indices(args.candidate_indices),
        actions=_parse_csv_tuple(args.actions, ACTION_NAMES),
        ordering_policies=_parse_csv_tuple(args.ordering_policies, ORDERING_POLICIES),
        baseline_iterations=args.baseline_iterations,
        polish_iterations=args.polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
