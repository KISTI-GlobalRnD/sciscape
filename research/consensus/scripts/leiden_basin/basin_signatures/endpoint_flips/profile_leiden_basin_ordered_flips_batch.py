#!/usr/bin/env python3
"""Batch ordered-flip basin profiles over selected field34/cc target pairs."""

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


import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

from sciscape.clustering.leiden_basin_profile import SCORING_POLICIES  # noqa: E402
from profile_leiden_basin_ordered_flips import (  # noqa: E402
    BEAM_ROWS_FILENAME,
    DEFAULT_OUTPUT_DIR as DEFAULT_SINGLE_OUTPUT_DIR,
    FRONTIER_ROWS_FILENAME,
    SUMMARY_FILENAME as SINGLE_SUMMARY_FILENAME,
    _best_beam_rows,
    _markdown_table,
    _parse_csv_tuple,
    run_profile,
)
from run_leiden_basin_transition_operator_pilot import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_LANDSCAPE_DIR,
    DEFAULT_VANILLA_DIR,
)
from analyze_leiden_basin_transition_minimal_pathway import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_MINIMAL_PATHWAY_DIR,
)

DEFAULT_TARGET_SELECTION = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "pathway_target_basin_selection_field34_cc/pathway_target_basin_selection_pairs.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SINGLE_OUTPUT_DIR.parent / (
    "pathway_ordered_flip_frontier_field34_cc_v1_cases"
)
CASE_ROWS_FILENAME = "ordered_flip_batch_case_rows.csv"
POLICY_ROWS_FILENAME = "ordered_flip_batch_policy_rows.csv"
SUMMARY_FILENAME = "ordered_flip_batch_summary.json"
REPORT_FILENAME = "ordered_flip_batch_report.md"

def selected_target_rows(
    target_rows: pd.DataFrame,
    *,
    max_priority: int,
    pair_ids: tuple[str, ...],
) -> pd.DataFrame:
    rows = target_rows.copy()
    rows["recommended_priority"] = pd.to_numeric(
        rows["recommended_priority"],
        errors="coerce",
    )
    rows = rows[rows["recommended_priority"].le(int(max_priority))].copy()
    if pair_ids:
        rows = rows[rows["pair_id"].astype(str).isin(set(pair_ids))].copy()
    return rows.sort_values(["recommended_priority", "pair_id"]).reset_index(drop=True)

def _fmt_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.6g}"

def _first_step_rows(frontier_rows: pd.DataFrame) -> pd.DataFrame:
    if frontier_rows.empty:
        return pd.DataFrame()
    first = frontier_rows[frontier_rows["step_index"].eq(1)].copy()
    rows: list[dict[str, Any]] = []
    for policy in SCORING_POLICIES:
        policy_first = first[first["scoring_policy"].eq(policy)]
        if policy_first.empty:
            continue
        score_col = f"{policy}_score"
        chosen = policy_first.sort_values(
            [score_col, "delta_q_immediate", "incremental_progress_fraction"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(
            {
                "scoring_policy": policy,
                "first_unit_id": chosen["unit_id"],
                "first_unit_node_count": int(chosen["unit_node_count"]),
                "first_delta_q_immediate": float(chosen["delta_q_immediate"]),
                "first_incremental_progress_fraction": float(
                    chosen["incremental_progress_fraction"]
                ),
                "first_raw_barrier_if_chosen": float(
                    chosen["raw_barrier_if_chosen"]
                ),
                "first_candidate_label": int(chosen["candidate_label"]),
                "first_vanilla_label": int(chosen["vanilla_label"]),
                "first_candidate_label_closure_extra_count": int(
                    chosen["candidate_label_closure_extra_count"]
                ),
            }
        )
    return pd.DataFrame(rows)

def summarize_profile_dir(
    *,
    profile_dir: Path,
    target_row: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary = json.loads((profile_dir / SINGLE_SUMMARY_FILENAME).read_text())
    frontier = pd.read_csv(profile_dir / FRONTIER_ROWS_FILENAME)
    beam = pd.read_csv(profile_dir / BEAM_ROWS_FILENAME)
    first_rows = _first_step_rows(frontier)
    best_rows = _best_beam_rows(beam)
    policy_rows = best_rows.merge(first_rows, on="scoring_policy", how="left")
    policy_rows["pair_id"] = summary["pair_id"]
    policy_rows["inspection_role"] = target_row.get("inspection_role", "")
    policy_rows["recommended_priority"] = int(target_row.get("recommended_priority", 0))
    front_columns = ["pair_id", "inspection_role", "recommended_priority"]
    policy_rows = policy_rows[
        front_columns + [c for c in policy_rows.columns if c not in front_columns]
    ]
    case_row = {
        "pair_id": summary["pair_id"],
        "inspection_role": target_row.get("inspection_role", ""),
        "recommended_priority": int(target_row.get("recommended_priority", 0)),
        "candidate_index": int(summary["candidate_index"]),
        "vanilla_seed": int(summary["vanilla_seed"]),
        "vanilla_randomness": float(summary["vanilla_randomness"]),
        "vanilla_minus_candidate_quality": float(
            summary["vanilla_minus_candidate_quality"]
        ),
        "candidate_support_size": int(summary["candidate_support_size"]),
        "vanilla_support_size": int(summary["vanilla_support_size"]),
        "v_only_support_size": int(summary["v_only_support_size"]),
        "unit_count": int(summary["unit_count"]),
        "frontier_rows": int(summary["frontier_rows"]),
        "beam_rows": int(summary["beam_rows"]),
        "best_final_support_distance_to_candidate": float(
            best_rows["result_support_distance_to_candidate"].min()
        ),
        "min_final_raw_barrier": float(best_rows["raw_barrier_so_far"].min()),
        "max_final_flipped_nodes": int(best_rows["flipped_node_count"].max()),
    }
    if not first_rows.empty:
        q_first = first_rows[first_rows["scoring_policy"].eq("q_first")]
        progress_first = first_rows[first_rows["scoring_policy"].eq("progress_first")]
        if not q_first.empty and not progress_first.empty:
            case_row["first_q_progress_same_unit"] = bool(
                str(q_first.iloc[0]["first_unit_id"])
                == str(progress_first.iloc[0]["first_unit_id"])
            )
            case_row["first_q_delta_q"] = float(
                q_first.iloc[0]["first_delta_q_immediate"]
            )
            case_row["first_q_progress"] = float(
                q_first.iloc[0]["first_incremental_progress_fraction"]
            )
            case_row["first_progress_delta_q"] = float(
                progress_first.iloc[0]["first_delta_q_immediate"]
            )
            case_row["first_progress_progress"] = float(
                progress_first.iloc[0]["first_incremental_progress_fraction"]
            )
    return case_row, policy_rows

def write_report(
    path: Path,
    *,
    case_rows: pd.DataFrame,
    policy_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Ordered Flip Basin Profile Batch",
        "",
        "This batch keeps the v0 unit definition fixed and expands only target cases.",
        "",
        "Scope: direction `V -> C`, unit type `label_intersection_block`, raw flips only.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "target_pair_count",
        "max_priority",
        "beam_width",
        "max_steps",
        "total_unit_rows",
        "total_frontier_rows",
        "total_beam_rows",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Case Rows", ""])
    case_cols = [
        "pair_id",
        "recommended_priority",
        "vanilla_minus_candidate_quality",
        "v_only_support_size",
        "unit_count",
        "first_q_progress_same_unit",
        "first_q_delta_q",
        "first_q_progress",
        "first_progress_delta_q",
        "first_progress_progress",
        "best_final_support_distance_to_candidate",
        "min_final_raw_barrier",
        "max_final_flipped_nodes",
    ]
    lines.extend(
        _markdown_table(
            case_rows[[c for c in case_cols if c in case_rows.columns]],
            max_rows=20,
        )
    )

    lines.extend(["", "## Policy End States", ""])
    policy_cols = [
        "pair_id",
        "scoring_policy",
        "step_index",
        "flipped_node_count",
        "delta_q_vs_start",
        "raw_barrier_so_far",
        "result_support_distance_to_candidate",
        "result_support_distance_to_vanilla",
        "first_unit_id",
        "first_delta_q_immediate",
        "first_incremental_progress_fraction",
    ]
    display = policy_rows.sort_values(["pair_id", "scoring_policy"])
    lines.extend(
        _markdown_table(display[[c for c in policy_cols if c in display.columns]], max_rows=40)
    )

    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- This batch tests whether the v0 split between QF-first and progress-first choices generalizes across target pairs.",
            "- It still uses raw upper-bound barriers. Do not read negative raw QF as an impossible transition without a later polish test.",
            "- If the first QF and progress units keep diverging, the next operator must explicitly price QF debt for basin progress.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_batch(
    *,
    target_selection: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    landscape_dir: Path,
    vanilla_dir: Path,
    minimal_pathway_dir: Path,
    max_priority: int,
    pair_ids: tuple[str, ...],
    baseline_iterations: int,
    polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    beam_width: int,
    max_steps: int,
    scoring_policies: tuple[str, ...],
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = selected_target_rows(
        pd.read_csv(target_selection),
        max_priority=max_priority,
        pair_ids=pair_ids,
    )
    if targets.empty:
        raise ValueError("No ordered-flip target rows selected")
    case_rows: list[dict[str, Any]] = []
    policy_frames: list[pd.DataFrame] = []
    for _, row in targets.iterrows():
        pair_id = str(row["pair_id"])
        profile_dir = output_dir / pair_id
        if force or not (profile_dir / SINGLE_SUMMARY_FILENAME).exists():
            run_profile(
                candidate_dirs=candidate_dirs,
                landscape_dir=landscape_dir,
                vanilla_dir=vanilla_dir,
                minimal_pathway_dir=minimal_pathway_dir,
                output_dir=profile_dir,
                candidate_index=int(row["candidate_index"]),
                vanilla_seed=int(row["vanilla_seed"]),
                vanilla_randomness=float(row["vanilla_randomness"]),
                vanilla_n=str(row["vanilla_requested_n_iterations"]),
                baseline_iterations=baseline_iterations,
                polish_iterations=polish_iterations,
                resolution=resolution,
                randomness=randomness,
                perturb_seed_offset=perturb_seed_offset,
                beam_width=beam_width,
                max_steps=max_steps,
                scoring_policies=scoring_policies,
            )
        case_row, policy_rows = summarize_profile_dir(
            profile_dir=profile_dir,
            target_row=row,
        )
        case_rows.append(case_row)
        policy_frames.append(policy_rows)
    case_frame = pd.DataFrame(case_rows).sort_values("recommended_priority")
    policy_frame = pd.concat(policy_frames, ignore_index=True)
    case_frame.to_csv(output_dir / CASE_ROWS_FILENAME, index=False)
    policy_frame.to_csv(output_dir / POLICY_ROWS_FILENAME, index=False)
    summary = {
        "schema": "leiden_basin_ordered_flip_profile_batch.v1",
        "target_selection": str(target_selection),
        "output_dir": str(output_dir),
        "target_pair_count": int(len(case_frame)),
        "max_priority": int(max_priority),
        "pair_ids": list(pair_ids),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "landscape_dir": str(landscape_dir),
        "vanilla_dir": str(vanilla_dir),
        "minimal_pathway_dir": str(minimal_pathway_dir),
        "beam_width": int(beam_width),
        "max_steps": int(max_steps),
        "scoring_policies": list(scoring_policies),
        "force": bool(force),
        "total_unit_rows": int(case_frame["unit_count"].sum()),
        "total_frontier_rows": int(case_frame["frontier_rows"].sum()),
        "total_beam_rows": int(case_frame["beam_rows"].sum()),
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        case_rows=case_frame,
        policy_rows=policy_frame,
        summary=summary,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-selection", type=Path, default=DEFAULT_TARGET_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument(
        "--minimal-pathway-dir",
        type=Path,
        default=DEFAULT_MINIMAL_PATHWAY_DIR,
    )
    parser.add_argument("--max-priority", type=int, default=4)
    parser.add_argument(
        "--pair-ids",
        default="",
        help="Optional comma-separated pair_id filter.",
    )
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--scoring-policies", default=",".join(SCORING_POLICIES))
    parser.add_argument("--force", action="store_true")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_batch(
        target_selection=args.target_selection,
        output_dir=args.output_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        landscape_dir=args.landscape_dir,
        vanilla_dir=args.vanilla_dir,
        minimal_pathway_dir=args.minimal_pathway_dir,
        max_priority=args.max_priority,
        pair_ids=_parse_csv_tuple(args.pair_ids, ()),
        baseline_iterations=args.baseline_iterations,
        polish_iterations=args.polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        scoring_policies=_parse_csv_tuple(args.scoring_policies, SCORING_POLICIES),
        force=bool(args.force),
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
