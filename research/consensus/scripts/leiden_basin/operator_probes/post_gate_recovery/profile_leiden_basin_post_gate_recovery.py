#!/usr/bin/env python3
"""Profile post-gate recovery behavior for selected tunneling detours."""

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

from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    POST_GATE_VERDICT_NEAR_MISS,
    POST_GATE_VERDICT_PLATEAU,
    POST_GATE_VERDICT_SUPPORT_TRADEOFF,
    annotate_post_gate_recovery_step_rows,
    summarize_post_gate_recovery_paths,
    trace_tunneling_path_states,
)

COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_RANK_DIR = COMBINED_DIR / "basin_transition_tunneling_path_rank_field34_cc_v0"
DEFAULT_STATE_DIR = COMBINED_DIR / "basin_transition_side_route_expansion_field34_cc_c0_v0"
DEFAULT_OUTPUT_DIR = COMBINED_DIR / "basin_transition_post_gate_recovery_field34_cc_c0_v0"

CANDIDATE_ROWS_FILENAME = "tunneling_path_candidate_rows.csv"
STATE_ROWS_FILENAME = "target_elbow_polish_states.csv"
FILTERED_CANDIDATES_FILENAME = "post_gate_recovery_candidate_rows.csv"
TRACE_ROWS_FILENAME = "post_gate_recovery_trace_rows.csv"
STEP_ROWS_FILENAME = "post_gate_recovery_step_rows.csv"
PATH_SUMMARY_FILENAME = "post_gate_recovery_path_summary_rows.csv"
PREFIX_SUMMARY_FILENAME = "post_gate_recovery_prefix_summary_rows.csv"
SUMMARY_FILENAME = "post_gate_recovery_summary.json"
CONFIG_FILENAME = "post_gate_recovery_config.json"
REPORT_FILENAME = "post_gate_recovery_report.md"

def _parse_prefix_ranks(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]

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

def _count_where(frame: pd.DataFrame, column: str, value: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].astype(str).eq(value).sum())

def _prefix_summary(summary_rows: pd.DataFrame) -> pd.DataFrame:
    if summary_rows.empty:
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    for prefix_rank, group in summary_rows.groupby("path_prefix_rank", sort=True):
        detours = group[group["tunnel_route_label"].astype(str).eq("unrecovered_detour")]
        source = detours if not detours.empty else group
        near_source = source[
            source["post_gate_verdict"].astype(str).eq(POST_GATE_VERDICT_NEAR_MISS)
        ]
        best_recovery_source = near_source if not near_source.empty else source
        best_recovery = best_recovery_source.sort_values(
            [
                "post_gate_best_delta_q_gain_from_gate",
                "post_gate_final_support",
                "post_gate_final_target_progress",
            ],
            ascending=[False, False, False],
        ).iloc[0]
        best_support = source.sort_values(
            [
                "post_gate_final_support",
                "post_gate_final_target_progress",
                "post_gate_final_delta_q",
            ],
            ascending=[False, False, False],
        ).iloc[0]
        out.append(
            {
                "path_prefix_rank": int(prefix_rank),
                "rows": int(len(group)),
                "detour_rows": int(len(detours)),
                "near_miss_rows": _count_where(
                    group,
                    "post_gate_verdict",
                    POST_GATE_VERDICT_NEAR_MISS,
                ),
                "support_tradeoff_rows": _count_where(
                    group,
                    "post_gate_verdict",
                    POST_GATE_VERDICT_SUPPORT_TRADEOFF,
                ),
                "plateau_rows": _count_where(
                    group,
                    "post_gate_verdict",
                    POST_GATE_VERDICT_PLATEAU,
                ),
                "best_delta_gain_from_gate": float(
                    group["post_gate_best_delta_q_gain_from_gate"].max()
                ),
                "best_final_delta_q": float(group["post_gate_final_delta_q"].max()),
                "best_final_support": float(group["post_gate_final_support"].max()),
                "best_final_target_progress": float(
                    group["post_gate_final_target_progress"].max()
                ),
                "best_recovery_state_id": best_recovery["path_final_state_id"],
                "best_recovery_verdict": best_recovery["post_gate_verdict"],
                "best_recovery_delta_gain": float(
                    best_recovery["post_gate_best_delta_q_gain_from_gate"]
                ),
                "best_support_state_id": best_support["path_final_state_id"],
                "best_support_verdict": best_support["post_gate_verdict"],
                "best_support_final_support": float(
                    best_support["post_gate_final_support"]
                ),
            }
        )
    return pd.DataFrame(out)

def write_report(
    path: Path,
    *,
    candidate_rows: pd.DataFrame,
    path_summary: pd.DataFrame,
    prefix_summary: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Post-Gate Recovery Profile",
        "",
        "This artifact focuses on p6/p8/p10 side-route detours after they cross the",
        "candidate-directed support gate.  It separates QF recovery trends from",
        "support-only deepening and plateau behavior.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "rank_dir",
        "state_dir",
        "output_dir",
        "candidate_rows",
        "trace_rows",
        "path_summary_rows",
        "near_miss_rows",
        "support_tradeoff_rows",
        "plateau_rows",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Prefix Verdict", ""])
    prefix_cols = [
        "path_prefix_rank",
        "detour_rows",
        "near_miss_rows",
        "support_tradeoff_rows",
        "plateau_rows",
        "best_delta_gain_from_gate",
        "best_final_delta_q",
        "best_final_support",
        "best_final_target_progress",
        "best_recovery_verdict",
        "best_support_verdict",
    ]
    lines.extend(
        _markdown_table(
            prefix_summary[[column for column in prefix_cols if column in prefix_summary]]
        )
    )

    lines.extend(["", "## Best Detour Rows", ""])
    detours = path_summary[
        path_summary["tunnel_route_label"].astype(str).eq("unrecovered_detour")
    ].copy()
    if not detours.empty:
        detours = detours.sort_values(
            [
                "post_gate_verdict",
                "post_gate_best_delta_q_gain_from_gate",
                "post_gate_final_support",
            ],
            ascending=[True, False, False],
        )
    detour_cols = [
        "path_prefix_rank",
        "path_selection_policy",
        "path_policy",
        "post_gate_verdict",
        "post_gate_gate_delta_q",
        "post_gate_best_delta_q",
        "post_gate_final_delta_q",
        "post_gate_best_delta_q_gain_from_gate",
        "post_gate_final_support",
        "post_gate_final_target_progress",
        "post_gate_recovery_step_count",
        "post_gate_support_deepening_step_count",
    ]
    lines.extend(_markdown_table(detours[[column for column in detour_cols if column in detours]]))

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `near_miss_recovery_trend` means QF improves after the gate but remains below start.",
            "- `support_deepening_quality_tradeoff` means later steps buy more support while losing QF from the best post-gate point.",
            "- `post_gate_plateau` means post-gate steps do not materially recover QF or deepen support.",
            "",
            "The rows are diagnostic, not an acceptance policy.  They identify where a",
            "future recovery operator should look before any Dongdaemun default is promoted.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_profile(
    *,
    rank_dir: Path,
    state_dir: Path,
    state_rows_filename: str,
    output_dir: Path,
    artifact_label: str,
    pair_id: str,
    prefix_ranks: list[int],
    support_gate: float,
    progress_margin: float,
    min_q_recovery_gain: float,
    min_support_gain: float,
    min_progress_gain: float,
) -> dict[str, Any]:
    candidates = pd.read_csv(rank_dir / CANDIDATE_ROWS_FILENAME)
    states = pd.read_csv(state_dir / state_rows_filename)
    filtered = candidates[
        candidates["artifact_label"].astype(str).eq(artifact_label)
        & candidates["pair_id"].astype(str).eq(pair_id)
        & candidates["path_prefix_rank"].astype(int).isin(prefix_ranks)
    ].copy()
    if filtered.empty:
        raise ValueError("No candidate rows matched the requested filter")

    trace = trace_tunneling_path_states(
        filtered,
        state_rows=states,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    step_rows = annotate_post_gate_recovery_step_rows(
        trace,
        min_q_recovery_gain=min_q_recovery_gain,
        min_support_gain=min_support_gain,
        min_progress_gain=min_progress_gain,
    )
    path_summary = summarize_post_gate_recovery_paths(
        trace,
        min_q_recovery_gain=min_q_recovery_gain,
        min_support_gain=min_support_gain,
        min_progress_gain=min_progress_gain,
    )
    prefix_summary = _prefix_summary(path_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_dir / FILTERED_CANDIDATES_FILENAME, index=False)
    trace.to_csv(output_dir / TRACE_ROWS_FILENAME, index=False)
    step_rows.to_csv(output_dir / STEP_ROWS_FILENAME, index=False)
    path_summary.to_csv(output_dir / PATH_SUMMARY_FILENAME, index=False)
    prefix_summary.to_csv(output_dir / PREFIX_SUMMARY_FILENAME, index=False)

    summary = {
        "schema": "leiden_basin_post_gate_recovery_profile.v0",
        "rank_dir": str(rank_dir),
        "state_dir": str(state_dir),
        "state_rows_filename": state_rows_filename,
        "output_dir": str(output_dir),
        "artifact_label": artifact_label,
        "pair_id": pair_id,
        "prefix_ranks": prefix_ranks,
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
        "min_q_recovery_gain": float(min_q_recovery_gain),
        "min_support_gain": float(min_support_gain),
        "min_progress_gain": float(min_progress_gain),
        "candidate_rows": int(len(filtered)),
        "trace_rows": int(len(trace)),
        "step_rows": int(len(step_rows)),
        "path_summary_rows": int(len(path_summary)),
        "prefix_summary_rows": int(len(prefix_summary)),
        "near_miss_rows": _count_where(
            path_summary,
            "post_gate_verdict",
            POST_GATE_VERDICT_NEAR_MISS,
        ),
        "support_tradeoff_rows": _count_where(
            path_summary,
            "post_gate_verdict",
            POST_GATE_VERDICT_SUPPORT_TRADEOFF,
        ),
        "plateau_rows": _count_where(
            path_summary,
            "post_gate_verdict",
            POST_GATE_VERDICT_PLATEAU,
        ),
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config = {
        "rank_dir": str(rank_dir),
        "state_dir": str(state_dir),
        "state_rows_filename": state_rows_filename,
        "output_dir": str(output_dir),
        "artifact_label": artifact_label,
        "pair_id": pair_id,
        "prefix_ranks": prefix_ranks,
        "support_gate": support_gate,
        "progress_margin": progress_margin,
        "min_q_recovery_gain": min_q_recovery_gain,
        "min_support_gain": min_support_gain,
        "min_progress_gain": min_progress_gain,
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        candidate_rows=filtered,
        path_summary=path_summary,
        prefix_summary=prefix_summary,
        summary=summary,
    )
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-dir", type=Path, default=DEFAULT_RANK_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--state-rows-filename", default=STATE_ROWS_FILENAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-label", default="side_route_expansion_c0")
    parser.add_argument("--pair-id", default="c0-s11-r0.001")
    parser.add_argument("--prefix-ranks", default="6,8,10")
    parser.add_argument("--support-gate", type=float, default=0.05)
    parser.add_argument("--progress-margin", type=float, default=0.005)
    parser.add_argument("--min-q-recovery-gain", type=float, default=1e-9)
    parser.add_argument("--min-support-gain", type=float, default=1e-9)
    parser.add_argument("--min-progress-gain", type=float, default=1e-9)
    args = parser.parse_args()

    summary = run_profile(
        rank_dir=args.rank_dir,
        state_dir=args.state_dir,
        state_rows_filename=args.state_rows_filename,
        output_dir=args.output_dir,
        artifact_label=args.artifact_label,
        pair_id=args.pair_id,
        prefix_ranks=_parse_prefix_ranks(args.prefix_ranks),
        support_gate=args.support_gate,
        progress_margin=args.progress_margin,
        min_q_recovery_gain=args.min_q_recovery_gain,
        min_support_gain=args.min_support_gain,
        min_progress_gain=args.min_progress_gain,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
