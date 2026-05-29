#!/usr/bin/env python3
"""Pilot label-internal repair for high-ratio closure labels.

This diagnostic asks whether a large candidate closure label can be improved by
splitting it internally under a bounded local polish. It is intentionally
separate from the shrink-from-vanilla family and is not a production policy.
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
    NODE_ROWS_FILENAME,
    boundary_anchor_nodes,
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
)
from rank_leiden_basin_transition_closure_frontier import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_FRONTIER_DIR,
    FRONTIER_ROWS_FILENAME,
    PAIR_COLUMNS,
)
from run_leiden_basin_transition_closure_operator_pilot import (  # noqa: E402
    _control_rows_for_pair,
    _evaluate_result,
    _pair_mask,
    _safe_float,
    _score_membership,
    _truthy_series,
    direct_nodes_for_frontier_row,
    split_nodes_to_fresh_donor_labels,
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
    "basin_transition_label_internal_repair_pilot_field34_cc"
)
ROWS_FILENAME = "basin_transition_label_internal_repair_rows.csv"
SUMMARY_FILENAME = "basin_transition_label_internal_repair_summary.json"
REPORT_FILENAME = "basin_transition_label_internal_repair_report.md"

RAW_OPERATOR = "label_internal_vanilla_seed_raw"
POLISH_OPERATOR = "label_internal_vanilla_seed_polish"
REPAIR_OPERATOR_NAMES = (RAW_OPERATOR, POLISH_OPERATOR)

def selected_repair_labels(
    frontier_rows: pd.DataFrame,
    *,
    closure_mode: str,
    max_pairs: int,
    max_labels_per_pair: int,
    min_closure_context_ratio: float,
    max_closure_nodes: int,
) -> pd.DataFrame:
    """Select high-ratio frontier labels for internal repair."""
    if frontier_rows.empty:
        return frontier_rows.copy()
    rows = frontier_rows[
        _truthy_series(frontier_rows["frontier_selected"])
        & frontier_rows["closure_mode"].astype(str).eq(str(closure_mode))
    ].copy()
    if rows.empty:
        return rows
    rows["closure_context_ratio"] = pd.to_numeric(
        rows["closure_context_ratio"],
        errors="coerce",
    )
    rows["closure_node_count"] = pd.to_numeric(
        rows["closure_node_count"],
        errors="coerce",
    )
    rows["frontier_score"] = pd.to_numeric(rows["frontier_score"], errors="coerce")
    rows = rows[
        rows["closure_context_ratio"].ge(float(min_closure_context_ratio))
        & rows["closure_node_count"].le(int(max_closure_nodes))
    ].copy()
    if rows.empty:
        return rows
    rows = rows.sort_values(
        [*PAIR_COLUMNS, "closure_context_ratio", "frontier_score"],
        ascending=[*([True] * len(PAIR_COLUMNS)), False, False],
    )
    rows = rows.groupby(PAIR_COLUMNS, dropna=False).head(int(max_labels_per_pair))
    pair_keys = rows[PAIR_COLUMNS].drop_duplicates().head(int(max_pairs))
    keep = np.zeros(len(rows), dtype=np.bool_)
    for _, pair in pair_keys.iterrows():
        keep |= _pair_mask(rows, pair)
    return rows[keep].reset_index(drop=True)

def closure_nodes_for_label(membership: np.ndarray, label: int) -> np.ndarray:
    return np.flatnonzero(
        np.asarray(membership, dtype=np.uint64) == np.uint64(int(label))
    ).astype(np.uint32, copy=False)

def mutable_nodes_for_label_repair(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    closure_nodes: np.ndarray,
    direct_nodes: np.ndarray,
    max_boundary_anchors: int,
) -> tuple[np.ndarray, int]:
    anchors, truncated = boundary_anchor_nodes(
        src=src,
        dst=dst,
        support_nodes=np.union1d(closure_nodes, direct_nodes),
        max_anchors=max_boundary_anchors,
    )
    mutable = np.union1d(closure_nodes, anchors).astype(np.uint32, copy=False)
    return mutable, int(truncated)

def split_closure_by_donor(
    *,
    membership: np.ndarray,
    donor_membership: np.ndarray,
    closure_nodes: np.ndarray,
) -> np.ndarray:
    repaired, _mapping, _next_label = split_nodes_to_fresh_donor_labels(
        membership,
        donor_membership,
        closure_nodes,
    )
    return _compact_membership(repaired)

def _repair_release_stats(
    *,
    row: pd.Series,
    direct_nodes: np.ndarray,
    closure_nodes: np.ndarray,
    mutable_nodes: np.ndarray,
    truncated_boundary_anchors: int,
    fixed_outside_mutable: bool,
) -> dict[str, Any]:
    return {
        "closure_label": int(row["closure_label"]),
        "closure_context_ratio": _safe_float(row.get("closure_context_ratio")),
        "frontier_score": _safe_float(row.get("frontier_score")),
        "direct_node_count": int(direct_nodes.size),
        "closure_node_count": int(closure_nodes.size),
        "mutable_node_count": int(mutable_nodes.size),
        "truncated_boundary_anchor_count": int(truncated_boundary_anchors),
        "donor_split": "vanilla_labels",
        "fixed_outside_mutable": bool(fixed_outside_mutable),
    }

def diagnostic_label_for_repair_row(
    row: pd.Series,
    *,
    material_delta: float = 1e-9,
    min_support_shift_from_candidate: float = 0.1,
) -> str:
    if str(row["operator"]) not in REPAIR_OPERATOR_NAMES:
        return "control"
    if float(row["delta_vs_candidate"]) < -float(material_delta):
        return "quality_loss"
    if (
        float(row["delta_vs_vanilla"]) < -float(material_delta)
        or float(row["delta_vs_control_extra"]) < -float(material_delta)
    ):
        return "seed_control_dominates"
    if (
        float(row["result_support_distance_to_candidate"])
        < float(min_support_shift_from_candidate)
    ):
        return "quality_win_same_basin"
    return "quality_win_support_shift"

def _operator_rows_for_label(
    *,
    graph: Any,
    arrays: Any,
    baseline: Any,
    candidate: Any,
    vanilla: Any,
    label_row: pd.Series,
    node_rows: pd.DataFrame,
    candidate_support: np.ndarray,
    vanilla_support: np.ndarray,
    sketch_nodes: np.ndarray,
    context: dict[str, Any],
    resolution: float,
    randomness: float,
    local_polish_iterations: int,
    operator_seed: int,
    max_boundary_anchors: int,
) -> list[dict[str, Any]]:
    direct_nodes = direct_nodes_for_frontier_row(
        node_rows=node_rows,
        frontier_row=label_row,
    )
    closure_nodes = closure_nodes_for_label(
        candidate.recreated.membership,
        int(label_row["closure_label"]),
    )
    if closure_nodes.size == 0:
        return []
    mutable_nodes, truncated = mutable_nodes_for_label_repair(
        src=arrays.src,
        dst=arrays.dst,
        closure_nodes=closure_nodes,
        direct_nodes=direct_nodes,
        max_boundary_anchors=max_boundary_anchors,
    )
    seeded = split_closure_by_donor(
        membership=candidate.recreated.membership,
        donor_membership=vanilla.membership,
        closure_nodes=closure_nodes,
    )
    raw_stats = _repair_release_stats(
        row=label_row,
        direct_nodes=direct_nodes,
        closure_nodes=closure_nodes,
        mutable_nodes=mutable_nodes,
        truncated_boundary_anchors=truncated,
        fixed_outside_mutable=False,
    )
    raw = _score_membership(graph, seeded, resolution=resolution)
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
    fixed = fixed_outside(int(seeded.size), mutable_nodes)
    polished = _run_leiden(
        graph,
        resolution=resolution,
        seed=operator_seed + int(label_row["closure_label"]),
        n_iterations=local_polish_iterations,
        randomness=randomness,
        initial_membership=seeded,
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
    frontier_dir: Path,
    boundary_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    output_dir: Path,
    closure_mode: str,
    max_pairs: int,
    max_labels_per_pair: int,
    min_closure_context_ratio: float,
    max_closure_nodes: int,
    max_boundary_anchors: int,
    baseline_iterations: int,
    transition_iterations: int,
    polish_iterations: int,
    local_polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier_rows = pd.read_csv(frontier_dir / FRONTIER_ROWS_FILENAME)
    selected = selected_repair_labels(
        frontier_rows,
        closure_mode=closure_mode,
        max_pairs=max_pairs,
        max_labels_per_pair=max_labels_per_pair,
        min_closure_context_ratio=min_closure_context_ratio,
        max_closure_nodes=max_closure_nodes,
    )
    if selected.empty:
        raise ValueError("No label-internal repair frontier rows selected")
    node_rows = pd.read_csv(boundary_dir / NODE_ROWS_FILENAME)
    candidates = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)

    baseline_cache: dict[str, Any] = {}
    candidate_cache: dict[tuple[str, int], Any] = {}
    vanilla_cache: dict[tuple[str, int, float, str], Any] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, Any]] = {}
    out_rows: list[dict[str, Any]] = []
    pair_rows = selected[PAIR_COLUMNS].drop_duplicates().reset_index(drop=True)

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
            "closure_mode": closure_mode,
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
        pair_labels = selected[_pair_mask(selected, pair)]
        for _, label_row in pair_labels.iterrows():
            out_rows.extend(
                _operator_rows_for_label(
                    graph=graph,
                    arrays=arrays,
                    baseline=baseline,
                    candidate=candidate,
                    vanilla=vanilla,
                    label_row=label_row,
                    node_rows=node_rows,
                    candidate_support=candidate_support,
                    vanilla_support=vanilla_support,
                    sketch_nodes=sketch_nodes,
                    context=context,
                    resolution=resolution,
                    randomness=randomness,
                    local_polish_iterations=local_polish_iterations,
                    operator_seed=operator_seed,
                    max_boundary_anchors=max_boundary_anchors,
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
            diagnostic_label_for_repair_row(row) for _, row in rows.iterrows()
        ]
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    repair_rows = rows[rows["operator"].isin(REPAIR_OPERATOR_NAMES)]
    summary = {
        "schema": "leiden_basin_transition_label_internal_repair_pilot.v1",
        "frontier_dir": str(frontier_dir),
        "boundary_dir": str(boundary_dir),
        "vanilla_dir": str(vanilla_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "output_dir": str(output_dir),
        "closure_mode": closure_mode,
        "selected_pair_count": int(len(pair_rows)),
        "selected_label_rows": int(len(selected)),
        "operator_rows": int(len(rows)),
        "repair_operator_rows": int(len(repair_rows)),
        "min_closure_context_ratio": float(min_closure_context_ratio),
        "max_closure_nodes": int(max_closure_nodes),
        "max_boundary_anchors": int(max_boundary_anchors),
        "baseline_iterations": int(baseline_iterations),
        "transition_iterations": int(transition_iterations),
        "polish_iterations": int(polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
    }
    if not repair_rows.empty:
        summary["best_repair_delta_vs_candidate"] = float(
            repair_rows["delta_vs_candidate"].max()
        )
        summary["best_repair_delta_vs_vanilla"] = float(
            repair_rows["delta_vs_vanilla"].max()
        )
        summary["best_repair_delta_vs_control_extra"] = float(
            repair_rows["delta_vs_control_extra"].max()
        )
        summary["max_repair_support_shift_from_candidate"] = float(
            repair_rows["result_support_distance_to_candidate"].max()
        )
        summary["min_repair_support_distance_to_vanilla"] = float(
            repair_rows["result_support_distance_to_vanilla"].min()
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
        "# Label Internal Repair Pilot",
        "",
        "This diagnostic splits high-ratio candidate closure labels by vanilla labels and runs bounded local repair.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "selected_pair_count",
        "selected_label_rows",
        "operator_rows",
        "repair_operator_rows",
        "best_repair_delta_vs_candidate",
        "best_repair_delta_vs_vanilla",
        "best_repair_delta_vs_control_extra",
        "max_repair_support_shift_from_candidate",
        "min_repair_support_distance_to_vanilla",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Operator Summary", ""])
    if not rows.empty:
        operator_summary = (
            rows.groupby("operator", as_index=False)
            .agg(
                rows=("operator", "size"),
                delta_vs_candidate_max=("delta_vs_candidate", "max"),
                delta_vs_candidate_median=("delta_vs_candidate", "median"),
                delta_vs_vanilla_max=("delta_vs_vanilla", "max"),
                delta_vs_vanilla_median=("delta_vs_vanilla", "median"),
                delta_vs_control_extra_max=("delta_vs_control_extra", "max"),
                delta_vs_control_extra_median=("delta_vs_control_extra", "median"),
                support_distance_to_candidate_median=(
                    "result_support_distance_to_candidate",
                    "median",
                ),
                support_distance_to_vanilla_median=(
                    "result_support_distance_to_vanilla",
                    "median",
                ),
                closure_nodes_median=("closure_node_count", "median"),
                mutable_nodes_median=("mutable_node_count", "median"),
                elapsed_sec_median=("elapsed_sec", "median"),
            )
            .sort_values("operator")
        )
        lines.extend(_markdown_table(operator_summary, max_rows=40))
    lines.extend(["", "## Repair Rows", ""])
    repair_rows = rows[rows["operator"].isin(REPAIR_OPERATOR_NAMES)]
    display_cols = [
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "operator",
        "closure_label",
        "closure_context_ratio",
        "direct_node_count",
        "closure_node_count",
        "mutable_node_count",
        "delta_vs_candidate",
        "delta_vs_vanilla",
        "delta_vs_control_extra",
        "diagnostic_label",
        "result_support_distance_to_candidate",
        "result_support_distance_to_vanilla",
        "elapsed_sec",
    ]
    lines.extend(
        _markdown_table(
            repair_rows[[c for c in display_cols if c in repair_rows.columns]],
            max_rows=60,
        )
    )
    lines.extend(["", "## Diagnostic Labels", ""])
    if not repair_rows.empty:
        labels = (
            repair_rows.groupby(["operator", "diagnostic_label"], as_index=False)
            .agg(rows=("operator", "size"))
            .sort_values(["operator", "diagnostic_label"])
        )
        lines.extend(_markdown_table(labels, max_rows=60))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- This tests label-internal repair, not a shrink threshold sweep.",
            "- A useful row must beat candidate, vanilla, and control while moving support meaningfully.",
            "- Rows that only split labels without quality/cost benefit remain diagnostic.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dir", type=Path, default=DEFAULT_FRONTIER_DIR)
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--closure-mode", default="candidate_label")
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--max-labels-per-pair", type=int, default=2)
    parser.add_argument("--min-closure-context-ratio", type=float, default=20.0)
    parser.add_argument("--max-closure-nodes", type=int, default=300)
    parser.add_argument("--max-boundary-anchors", type=int, default=64)
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
        frontier_dir=args.frontier_dir,
        boundary_dir=args.boundary_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dir=args.vanilla_dir,
        output_dir=args.output_dir,
        closure_mode=args.closure_mode,
        max_pairs=args.max_pairs,
        max_labels_per_pair=args.max_labels_per_pair,
        min_closure_context_ratio=args.min_closure_context_ratio,
        max_closure_nodes=args.max_closure_nodes,
        max_boundary_anchors=args.max_boundary_anchors,
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
