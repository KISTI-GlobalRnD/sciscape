#!/usr/bin/env python3
"""Analyze barrier-aware non-greedy prefixes from ordered-flip profiles."""

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

from sciscape.clustering.leiden_basin_profile import (  # noqa: E402
    BARRIER_FAILURE_LABELS,
    annotate_barrier_aware_prefixes,
    select_barrier_progress_frontier,
)
from profile_leiden_basin_ordered_flips import (  # noqa: E402
    BEAM_ROWS_FILENAME,
    FRONTIER_ROWS_FILENAME,
    SUMMARY_FILENAME as SINGLE_SUMMARY_FILENAME,
    _markdown_table,
)
from profile_leiden_basin_ordered_flips_batch import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PROFILE_BATCH_DIR,
)

DEFAULT_OUTPUT_DIR = DEFAULT_PROFILE_BATCH_DIR.parent / (
    "pathway_barrier_aware_prefix_field34_cc_v1"
)
PREFIX_ROWS_FILENAME = "barrier_aware_prefix_rows.csv"
CASE_ROWS_FILENAME = "barrier_aware_case_rows.csv"
SUMMARY_FILENAME = "barrier_aware_summary.json"
REPORT_FILENAME = "barrier_aware_report.md"

def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())

def _labels_contain(labels: pd.Series, label: str) -> pd.Series:
    return labels.astype(str).str.split(";").apply(lambda parts: label in set(parts))

def _profile_dirs(input_dir: Path, pair_ids: tuple[str, ...]) -> list[Path]:
    candidates = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_dir() and (path / SINGLE_SUMMARY_FILENAME).exists()
    ]
    if pair_ids:
        allowed = set(pair_ids)
        candidates = [path for path in candidates if path.name in allowed]
    return candidates

def summarize_case(
    *,
    profile_dir: Path,
    max_prefix_rows: int,
    min_support_progress: float,
    barrier_floor: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary = json.loads((profile_dir / SINGLE_SUMMARY_FILENAME).read_text())
    frontier = pd.read_csv(profile_dir / FRONTIER_ROWS_FILENAME)
    beam = pd.read_csv(profile_dir / BEAM_ROWS_FILENAME)
    annotated = annotate_barrier_aware_prefixes(
        frontier_rows=frontier,
        beam_rows=beam,
        v_only_support_size=int(summary["v_only_support_size"]),
        barrier_floor=barrier_floor,
    )
    selected = select_barrier_progress_frontier(
        annotated,
        max_rows=max_prefix_rows,
        min_support_progress=min_support_progress,
    ).copy()
    for key in [
        "case",
        "field",
        "method",
        "pair_id",
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "vanilla_requested_n_iterations",
        "candidate_support_size",
        "vanilla_support_size",
        "v_only_support_size",
        "vanilla_minus_candidate_quality",
    ]:
        selected[key] = summary.get(key, "")
    front_cols = [
        "case",
        "field",
        "method",
        "pair_id",
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "vanilla_requested_n_iterations",
    ]
    selected = selected[front_cols + [c for c in selected.columns if c not in front_cols]]
    labels = annotated["greedy_failure_labels"]
    q_miss = int(_labels_contain(labels, "q_greedy_miss").sum())
    progress_miss = int(_labels_contain(labels, "progress_greedy_miss").sum())
    closure_miss = int(_labels_contain(labels, "closure_compound_miss").sum())
    recovery_miss = int(_labels_contain(labels, "polish_recovery_miss").sum())
    best = selected.iloc[0] if not selected.empty else None
    case_row: dict[str, Any] = {
        "pair_id": summary["pair_id"],
        "case": summary["case"],
        "field": summary.get("field", ""),
        "method": summary.get("method", ""),
        "candidate_index": int(summary["candidate_index"]),
        "vanilla_seed": int(summary["vanilla_seed"]),
        "vanilla_randomness": float(summary["vanilla_randomness"]),
        "v_only_support_size": int(summary["v_only_support_size"]),
        "total_frontier_rows": int(len(annotated)),
        "selected_prefix_rows": int(len(selected)),
        "q_greedy_miss_rows": q_miss,
        "progress_greedy_miss_rows": progress_miss,
        "closure_compound_miss_rows": closure_miss,
        "polish_recovery_miss_rows": recovery_miss,
    }
    for label in BARRIER_FAILURE_LABELS:
        case_row[f"{label}_selected_rows"] = (
            int(_labels_contain(selected["greedy_failure_labels"], label).sum())
            if not selected.empty
            else 0
        )
    if best is not None:
        case_row.update(
            {
                "best_prefix_unit_ids": best["prefix_unit_ids"],
                "best_prefix_unit_count": int(best["prefix_unit_count"]),
                "best_barrier_aware_score": float(best["barrier_aware_score"]),
                "best_peak_raw_barrier": float(best["peak_raw_barrier"]),
                "best_support_progress_fraction": float(
                    best["support_progress_fraction"]
                ),
                "best_support_progress_per_barrier_floor": float(
                    best["support_progress_per_barrier_floor"]
                ),
                "best_prefix_flipped_node_count_estimate": int(
                    best["prefix_flipped_node_count_estimate"]
                ),
                "best_greedy_failure_labels": best["greedy_failure_labels"],
            }
        )
    return case_row, selected

def write_report(
    path: Path,
    *,
    case_rows: pd.DataFrame,
    prefix_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Barrier-Aware Pathway Prefixes",
        "",
        "This diagnostic re-scores existing ordered-flip frontier rows. It does not rerun Leiden and does not accept an operator.",
        "",
        "Goal: find non-greedy prefixes that may cross a basin wall with bounded raw barrier and enough support progress to justify a later polish-recovery test.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "input_dir",
        "profile_count",
        "total_selected_prefix_rows",
        "max_prefix_rows_per_case",
        "min_support_progress",
        "barrier_floor",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Case Rows", ""])
    case_cols = [
        "pair_id",
        "v_only_support_size",
        "total_frontier_rows",
        "selected_prefix_rows",
        "q_greedy_miss_rows",
        "progress_greedy_miss_rows",
        "closure_compound_miss_rows",
        "polish_recovery_miss_rows",
        "best_prefix_unit_count",
        "best_peak_raw_barrier",
        "best_support_progress_fraction",
        "best_support_progress_per_barrier_floor",
        "best_greedy_failure_labels",
    ]
    lines.extend(
        _markdown_table(
            case_rows[[c for c in case_cols if c in case_rows.columns]],
            max_rows=40,
        )
    )
    lines.extend(["", "## Top Prefix Rows", ""])
    prefix_cols = [
        "pair_id",
        "scoring_policy",
        "parent_state_id",
        "unit_id",
        "prefix_unit_count",
        "prefix_flipped_node_count_estimate",
        "support_progress_fraction",
        "peak_raw_barrier",
        "support_progress_per_barrier_floor",
        "barrier_aware_score",
        "q_rank_within_parent",
        "progress_rank_within_parent",
        "greedy_failure_labels",
        "prefix_unit_ids",
    ]
    display = prefix_rows.sort_values(
        ["barrier_aware_score", "support_progress_fraction"],
        ascending=[False, False],
    )
    lines.extend(
        _markdown_table(display[[c for c in prefix_cols if c in display.columns]], max_rows=60)
    )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- Infinite progress-per-zero-barrier rows are reported with a finite barrier floor score so zero-barrier local edits do not automatically end the search.",
            "- `polish_recovery_miss` means the raw prefix needs a later polish check; it is not evidence that the prefix is recoverable.",
            "- A prefix is actionable only if a later bounded-polish evaluator recovers QF debt while retaining support movement away from the source basin.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_analysis(
    *,
    input_dir: Path,
    output_dir: Path,
    pair_ids: tuple[str, ...],
    max_prefix_rows_per_case: int,
    min_support_progress: float,
    barrier_floor: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dirs = _profile_dirs(input_dir, pair_ids)
    if not profile_dirs:
        raise ValueError(f"No profile directories found in {input_dir}")
    case_rows: list[dict[str, Any]] = []
    prefix_frames: list[pd.DataFrame] = []
    for profile_dir in profile_dirs:
        case_row, prefixes = summarize_case(
            profile_dir=profile_dir,
            max_prefix_rows=max_prefix_rows_per_case,
            min_support_progress=min_support_progress,
            barrier_floor=barrier_floor,
        )
        case_rows.append(case_row)
        prefix_frames.append(prefixes)
    case_frame = pd.DataFrame(case_rows).sort_values(["field", "method", "pair_id"])
    prefix_frame = pd.concat(prefix_frames, ignore_index=True)
    case_frame.to_csv(output_dir / CASE_ROWS_FILENAME, index=False)
    prefix_frame.to_csv(output_dir / PREFIX_ROWS_FILENAME, index=False)
    summary = {
        "schema": "leiden_basin_barrier_aware_pathways.v1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pair_ids": list(pair_ids),
        "profile_count": int(len(profile_dirs)),
        "total_selected_prefix_rows": int(len(prefix_frame)),
        "max_prefix_rows_per_case": int(max_prefix_rows_per_case),
        "min_support_progress": float(min_support_progress),
        "barrier_floor": float(barrier_floor),
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        case_rows=case_frame,
        prefix_rows=prefix_frame,
        summary=summary,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_PROFILE_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pair-ids",
        default="",
        help="Optional comma-separated pair_id filter.",
    )
    parser.add_argument("--max-prefix-rows-per-case", type=int, default=50)
    parser.add_argument("--min-support-progress", type=float, default=0.0)
    parser.add_argument("--barrier-floor", type=float, default=1.0)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pair_ids=_parse_csv_tuple(args.pair_ids),
        max_prefix_rows_per_case=args.max_prefix_rows_per_case,
        min_support_progress=args.min_support_progress,
        barrier_floor=args.barrier_floor,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
