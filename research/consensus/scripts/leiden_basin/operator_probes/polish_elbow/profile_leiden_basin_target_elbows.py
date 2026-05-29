#!/usr/bin/env python3
"""Profile pull-curve elbow candidates for staged target-node growth."""

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

from analyze_leiden_basin_barrier_aware_pathways import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PREFIX_DIR,
    PREFIX_ROWS_FILENAME as BARRIER_PREFIX_ROWS_FILENAME,
)
from evaluate_leiden_basin_polish_prefixes import select_prefix_rows  # noqa: E402
from profile_leiden_basin_ordered_flips import UNIT_ROWS_FILENAME  # noqa: E402
from profile_leiden_basin_ordered_flips_batch import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PROFILE_BATCH_DIR,
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
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
)
from sciscape.clustering.leiden_basin_profile import v_only_support_nodes  # noqa: E402
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    ACTION_REMAINING_TARGET_TOPK,
    TransitionAction,
    cap_context_count,
    make_child_state,
    make_prefix_state,
    node_csv,
    prefix_direct_nodes,
    remaining_target_elbow_summary,
    remaining_target_pull_frame,
    unique_sorted_u32,
)

DEFAULT_OUTPUT_DIR = DEFAULT_PROFILE_BATCH_DIR.parent / (
    "basin_transition_target_elbow_field34_cc_v0"
)
STAGE_ROWS_FILENAME = "target_elbow_stage_rows.csv"
CURVE_ROWS_FILENAME = "target_elbow_curve_rows.csv"
CASE_ROWS_FILENAME = "target_elbow_case_rows.csv"
SUMMARY_FILENAME = "target_elbow_summary.json"
CONFIG_FILENAME = "target_elbow_config.json"
REPORT_FILENAME = "target_elbow_report.md"

def _parse_csv_tuple(value: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())

def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> list[str]:
    if frame.empty:
        return []
    display = frame.head(max_rows)
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines

def _selected_k_for_policy(summary: dict[str, Any], path_policy: str) -> int:
    if path_policy == "fixed_cap":
        return int(summary["fixed_effective_k"])
    if path_policy == "guarded_elbow":
        return int(summary["guarded_elbow_k"])
    raise ValueError(f"Unsupported elbow path policy: {path_policy}")

def _profile_case(
    *,
    case_prefix_rows: pd.DataFrame,
    profile_batch_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    max_stages: int,
    path_policies: tuple[str, ...],
    target_action_multiplier: float,
    max_target_action_nodes: int,
    cumulative_fraction: float,
    min_score_fraction: float,
    min_gap_fraction: float,
    min_guarded_pull_fraction: float,
    top_curve_rows: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = case_prefix_rows.iloc[0]
    case = str(first["case"])
    pair_id = str(first["pair_id"])
    candidate_index = int(first["candidate_index"])
    vanilla_seed = int(first["vanilla_seed"])
    vanilla_randomness = float(first["vanilla_randomness"])
    vanilla_n = str(first["vanilla_requested_n_iterations"])
    profile_dir = profile_batch_dir / pair_id
    units = pd.read_csv(profile_dir / UNIT_ROWS_FILENAME)
    candidate_rows = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)
    candidate_row = _find_candidate_row(
        candidate_rows,
        case=case,
        candidate_index=candidate_index,
    )
    vanilla_row = _find_vanilla_row(
        vanilla_rows,
        case=case,
        seed=vanilla_seed,
        randomness=vanilla_randomness,
        n_iterations=vanilla_n,
    )
    graph_dir = Path(str(vanilla_row["graph_dir"]))
    graph, node_weights, arrays = _load_graph(graph_dir)
    baseline = _run_leiden(
        graph,
        resolution=resolution,
        seed=int(candidate_row.get("seed", 0)),
        n_iterations=baseline_iterations,
        randomness=randomness,
    )
    candidate = _recreate_candidate(
        graph=graph,
        arrays=arrays,
        node_weights=node_weights,
        baseline_membership=baseline.membership,
        baseline_quality=baseline.quality,
        row=candidate_row,
        resolution=resolution,
        randomness=randomness,
        perturb_seed_offset=perturb_seed_offset,
        polish_iterations=candidate_polish_iterations,
    )
    vanilla = _run_leiden(
        graph,
        resolution=resolution,
        seed=vanilla_seed,
        n_iterations=int(_safe_int(vanilla_n, baseline_iterations) or baseline_iterations),
        randomness=vanilla_randomness,
    )
    candidate_support, _, target_nodes = v_only_support_nodes(
        baseline.membership,
        candidate.recreated.membership,
        vanilla.membership,
    )
    src = np.asarray(arrays.src, dtype=np.uint32)
    dst = np.asarray(arrays.dst, dtype=np.uint32)
    weight = np.asarray(arrays.weight, dtype=np.float64)
    node_count = int(baseline.membership.size)
    stage_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for prefix_rank, (_, prefix_row) in enumerate(case_prefix_rows.iterrows(), start=1):
        direct_nodes = prefix_direct_nodes(units, prefix_row["prefix_unit_ids"])
        root = make_prefix_state(
            state_id=f"{pair_id}:p{prefix_rank}:elbow",
            prefix_rank=prefix_rank,
            prefix_unit_ids=str(prefix_row["prefix_unit_ids"]),
            membership=vanilla.membership,
            quality=float(vanilla.quality),
            direct_nodes=direct_nodes,
            target_nodes=target_nodes,
            action_nodes=direct_nodes,
            mutable_nodes=direct_nodes,
        )
        for path_policy in path_policies:
            state = root
            for stage_index in range(1, int(max_stages) + 1):
                anchor_count = int(unique_sorted_u32(state.action_nodes).size)
                covered_count = int(unique_sorted_u32(state.covered_target_nodes).size)
                fixed_k = cap_context_count(
                    direct_node_count=anchor_count,
                    context_multiplier=target_action_multiplier,
                    max_context_nodes=max_target_action_nodes,
                )
                frame = remaining_target_pull_frame(
                    state=state,
                    src=src,
                    dst=dst,
                    weight=weight,
                    node_count=node_count,
                )
                summary = remaining_target_elbow_summary(
                    frame,
                    fixed_k=fixed_k,
                    cumulative_fraction=cumulative_fraction,
                    min_score_fraction=min_score_fraction,
                    min_gap_fraction=min_gap_fraction,
                    min_guarded_pull_fraction=min_guarded_pull_fraction,
                )
                selected_k = _selected_k_for_policy(summary, path_policy)
                selected = (
                    np.asarray(frame.head(selected_k)["node"], dtype=np.uint32)
                    if selected_k > 0 and not frame.empty
                    else np.asarray([], dtype=np.uint32)
                )
                context = {
                    "case": case,
                    "field": first.get("field", ""),
                    "method": first.get("method", ""),
                    "pair_id": pair_id,
                    "candidate_index": candidate_index,
                    "vanilla_seed": vanilla_seed,
                    "vanilla_randomness": vanilla_randomness,
                    "vanilla_requested_n_iterations": vanilla_n,
                    "prefix_rank": prefix_rank,
                    "prefix_unit_ids": str(prefix_row["prefix_unit_ids"]),
                    "barrier_aware_score": float(prefix_row["barrier_aware_score"]),
                    "peak_raw_barrier_input": float(prefix_row["peak_raw_barrier"]),
                    "support_progress_fraction_input": float(
                        prefix_row["support_progress_fraction"]
                    ),
                    "path_policy": path_policy,
                    "stage_index": stage_index,
                    "anchor_node_count": anchor_count,
                    "covered_target_count": covered_count,
                    "selected_k": int(selected_k),
                    "selected_node_ids": node_csv(selected),
                    "target_node_count": int(unique_sorted_u32(target_nodes).size),
                    "candidate_support_size": int(candidate_support.size),
                }
                stage_rows.append({**context, **summary})
                top = frame.head(int(top_curve_rows)).copy()
                for _, curve_row in top.iterrows():
                    curve_rows.append(
                        {
                            **context,
                            "rank": int(curve_row["rank"]),
                            "node": int(curve_row["node"]),
                            "pull_score": float(curve_row["pull_score"]),
                            "cumulative_pull": float(curve_row["cumulative_pull"]),
                            "cumulative_pull_fraction": float(
                                curve_row["cumulative_pull_fraction"]
                            ),
                            "score_fraction_of_top": float(
                                curve_row["score_fraction_of_top"]
                            ),
                            "next_gap": float(curve_row["next_gap"]),
                            "next_gap_fraction_of_top": float(
                                curve_row["next_gap_fraction_of_top"]
                            ),
                        }
                    )
                if selected.size == 0:
                    break
                state = make_child_state(
                    parent=state,
                    action=TransitionAction(
                        action_type=ACTION_REMAINING_TARGET_TOPK,
                        action_params=f"diagnostic_path_policy={path_policy}",
                        context_nodes=np.asarray([], dtype=np.uint32),
                        action_nodes=selected,
                    ),
                    membership=state.membership,
                    quality=state.quality,
                    elapsed_sec=0.0,
                    child_index=stage_index,
                )
    return pd.DataFrame(stage_rows), pd.DataFrame(curve_rows)

def _case_rows(stage_rows: pd.DataFrame) -> pd.DataFrame:
    if stage_rows.empty:
        return pd.DataFrame()
    grouped = stage_rows.groupby(["pair_id", "path_policy"], sort=True)
    out = grouped.agg(
        stage_rows=("stage_index", "size"),
        target_node_count=("target_node_count", "max"),
        median_anchor_node_count=("anchor_node_count", "median"),
        median_remaining_count=("remaining_count", "median"),
        median_fixed_k=("fixed_effective_k", "median"),
        median_guarded_elbow_k=("guarded_elbow_k", "median"),
        median_gap_elbow_k=("gap_elbow_k", "median"),
        median_cumulative_elbow_k=("cumulative_elbow_k", "median"),
        median_score_floor_k=("score_floor_k", "median"),
        median_fixed_pull_fraction=("fixed_pull_fraction", "median"),
        median_guarded_elbow_pull_fraction=("guarded_elbow_pull_fraction", "median"),
        median_gap_drop_fraction=("gap_elbow_drop_fraction_of_top", "median"),
        max_gap_drop_fraction=("gap_elbow_drop_fraction_of_top", "max"),
    ).reset_index()
    smaller_rows: list[int] = []
    for _, group in grouped:
        smaller_rows.append(
            int(
                (
                    group["guarded_elbow_k"].astype(int)
                    < group["fixed_effective_k"].astype(int)
                ).sum()
            )
        )
    out["guarded_smaller_than_fixed_rows"] = smaller_rows
    out["guarded_smaller_than_fixed_fraction"] = (
        out["guarded_smaller_than_fixed_rows"] / out["stage_rows"].clip(lower=1)
    )
    return out

def write_report(
    path: Path,
    *,
    stage_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Basin Target Elbow Profile v0",
        "",
        "This artifact profiles cheap pull-curve cut points for node-level staged target growth.",
        "",
        "It does not run bounded polish. It compares fixed-cap selection with candidate elbow cut points over the same pull-ranked remaining target nodes.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "prefix_dir",
        "profile_batch_dir",
        "stage_rows",
        "curve_rows",
        "case_rows",
        "pair_ids",
        "top_prefixes_per_case",
        "max_stages",
        "target_action_multiplier",
        "max_target_action_nodes",
        "cumulative_fraction",
        "min_score_fraction",
        "min_gap_fraction",
        "min_guarded_pull_fraction",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Case Rows", ""])
    case_cols = [
        "pair_id",
        "path_policy",
        "stage_rows",
        "target_node_count",
        "median_anchor_node_count",
        "median_remaining_count",
        "median_fixed_k",
        "median_guarded_elbow_k",
        "median_gap_elbow_k",
        "median_cumulative_elbow_k",
        "median_score_floor_k",
        "median_fixed_pull_fraction",
        "median_guarded_elbow_pull_fraction",
        "guarded_smaller_than_fixed_rows",
        "guarded_smaller_than_fixed_fraction",
        "median_gap_drop_fraction",
        "max_gap_drop_fraction",
    ]
    lines.extend(
        _markdown_table(case_rows[[c for c in case_cols if c in case_rows.columns]])
    )
    lines.extend(["", "## Smallest Guarded K Rows", ""])
    row_cols = [
        "pair_id",
        "path_policy",
        "prefix_rank",
        "stage_index",
        "anchor_node_count",
        "remaining_count",
        "fixed_effective_k",
        "guarded_elbow_k",
        "guarded_elbow_reason",
        "gap_elbow_k",
        "cumulative_elbow_k",
        "score_floor_k",
        "fixed_pull_fraction",
        "guarded_elbow_pull_fraction",
        "gap_elbow_drop_fraction_of_top",
        "selected_node_ids",
    ]
    if not stage_rows.empty:
        smallest = stage_rows.sort_values(
            ["guarded_elbow_k", "gap_elbow_drop_fraction_of_top"],
            ascending=[True, False],
        )
        lines.extend(
            _markdown_table(
                smallest[[c for c in row_cols if c in smallest.columns]],
                max_rows=40,
            )
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- A small elbow k is not an acceptance criterion.",
            "- The next validation must compare bounded-polish rows against fixed-cap top-k using material support shift, QF recovery, mutable-node cost, and runtime.",
            "- If guarded elbow only reduces k but loses support shift, it is a speed heuristic rather than a better basin-transition mechanism.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_profile(
    *,
    prefix_dir: Path,
    profile_batch_dir: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    pair_ids: tuple[str, ...],
    top_prefixes_per_case: int,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    max_stages: int,
    path_policies: tuple[str, ...],
    target_action_multiplier: float,
    max_target_action_nodes: int,
    cumulative_fraction: float,
    min_score_fraction: float,
    min_gap_fraction: float,
    min_guarded_pull_fraction: float,
    top_curve_rows: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = select_prefix_rows(
        pd.read_csv(prefix_dir / BARRIER_PREFIX_ROWS_FILENAME),
        pair_ids=pair_ids,
        top_prefixes_per_case=top_prefixes_per_case,
    )
    if prefixes.empty:
        raise ValueError("No prefix rows selected for target elbow profiling")
    stage_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for _, case_prefixes in prefixes.groupby("pair_id", sort=True):
        stage_rows, curve_rows = _profile_case(
            case_prefix_rows=case_prefixes,
            profile_batch_dir=profile_batch_dir,
            candidate_dirs=candidate_dirs,
            vanilla_dir=vanilla_dir,
            baseline_iterations=baseline_iterations,
            candidate_polish_iterations=candidate_polish_iterations,
            max_stages=max_stages,
            path_policies=path_policies,
            target_action_multiplier=target_action_multiplier,
            max_target_action_nodes=max_target_action_nodes,
            cumulative_fraction=cumulative_fraction,
            min_score_fraction=min_score_fraction,
            min_gap_fraction=min_gap_fraction,
            min_guarded_pull_fraction=min_guarded_pull_fraction,
            top_curve_rows=top_curve_rows,
            resolution=resolution,
            randomness=randomness,
            perturb_seed_offset=perturb_seed_offset,
        )
        stage_frames.append(stage_rows)
        curve_frames.append(curve_rows)
    stage_rows = (
        pd.concat(stage_frames, ignore_index=True) if stage_frames else pd.DataFrame()
    )
    curve_rows = (
        pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    )
    case_rows = _case_rows(stage_rows)
    stage_rows.to_csv(output_dir / STAGE_ROWS_FILENAME, index=False)
    curve_rows.to_csv(output_dir / CURVE_ROWS_FILENAME, index=False)
    case_rows.to_csv(output_dir / CASE_ROWS_FILENAME, index=False)
    config = {
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "pair_ids": list(pair_ids),
        "top_prefixes_per_case": int(top_prefixes_per_case),
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "max_stages": int(max_stages),
        "path_policies": list(path_policies),
        "target_action_multiplier": float(target_action_multiplier),
        "max_target_action_nodes": int(max_target_action_nodes),
        "cumulative_fraction": float(cumulative_fraction),
        "min_score_fraction": float(min_score_fraction),
        "min_gap_fraction": float(min_gap_fraction),
        "min_guarded_pull_fraction": float(min_guarded_pull_fraction),
        "top_curve_rows": int(top_curve_rows),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "perturb_seed_offset": int(perturb_seed_offset),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "leiden_basin_target_elbow.v0",
        "output_dir": str(output_dir),
        "stage_rows": int(len(stage_rows)),
        "curve_rows": int(len(curve_rows)),
        "case_rows": int(len(case_rows)),
        **config,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        stage_rows=stage_rows,
        case_rows=case_rows,
        summary=summary,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--profile-batch-dir", type=Path, default=DEFAULT_PROFILE_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--pair-ids", default="c0-s11-r0.001,c2-s11-r0")
    parser.add_argument("--top-prefixes-per-case", type=int, default=10)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--candidate-polish-iterations", type=int, default=5)
    parser.add_argument("--max-stages", type=int, default=3)
    parser.add_argument("--path-policies", default="fixed_cap,guarded_elbow")
    parser.add_argument("--target-action-multiplier", type=float, default=0.5)
    parser.add_argument("--max-target-action-nodes", type=int, default=64)
    parser.add_argument("--cumulative-fraction", type=float, default=0.80)
    parser.add_argument("--min-score-fraction", type=float, default=0.05)
    parser.add_argument("--min-gap-fraction", type=float, default=0.25)
    parser.add_argument("--min-guarded-pull-fraction", type=float, default=0.50)
    parser.add_argument("--top-curve-rows", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_profile(
        prefix_dir=args.prefix_dir,
        profile_batch_dir=args.profile_batch_dir,
        output_dir=args.output_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dir=args.vanilla_dir,
        pair_ids=_parse_csv_tuple(args.pair_ids),
        top_prefixes_per_case=args.top_prefixes_per_case,
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        max_stages=args.max_stages,
        path_policies=_parse_csv_tuple(args.path_policies),
        target_action_multiplier=args.target_action_multiplier,
        max_target_action_nodes=args.max_target_action_nodes,
        cumulative_fraction=args.cumulative_fraction,
        min_score_fraction=args.min_score_fraction,
        min_gap_fraction=args.min_gap_fraction,
        min_guarded_pull_fraction=args.min_guarded_pull_fraction,
        top_curve_rows=args.top_curve_rows,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
