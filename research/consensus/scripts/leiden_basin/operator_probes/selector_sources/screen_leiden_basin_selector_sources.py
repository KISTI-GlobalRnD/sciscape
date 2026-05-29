#!/usr/bin/env python3
"""Screen post-gate source states before expensive local-selector replay.

This diagnostic rebuilds post-gate source states, scores attachment-margin
handles against the source path's target-action context, and summarizes whether
the source has enough local handles to justify a full selector replay.  It does
not run recovery moves or selector-polish trials.
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

from analyze_leiden_basin_barrier_aware_pathways import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PREFIX_DIR,
    PREFIX_ROWS_FILENAME as BARRIER_PREFIX_ROWS_FILENAME,
)
from evaluate_leiden_basin_polish_prefixes import select_prefix_rows  # noqa: E402
from evaluate_leiden_basin_target_elbow_polish import (  # noqa: E402
    _rank_and_filter_prefix_rows,
)
from probe_leiden_basin_post_gate_recovery_moves import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_PROFILE_BATCH_DIR,
    DEFAULT_VANILLA_DIR,
    POST_GATE_PATH_SUMMARY_FILENAME,
    _load_case_context,
    _load_json,
    _markdown_table,
    _parse_node_ids,
    _prefix_context,
    _recorded_path_to_state,
    _replay_to_source_state,
)
from profile_leiden_basin_gate_attachment_candidates import _candidate_rows  # noqa: E402
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    ACTION_BOUNDARY_SHELL_TOPK,
    ACTION_CANDIDATE_CLOSURE_TOPK,
    ACTION_VANILLA_CLOSURE_TOPK,
    POST_GATE_VERDICT_NEAR_MISS,
    POST_GATE_VERDICT_PLATEAU,
    POST_GATE_VERDICT_SUPPORT_TRADEOFF,
    build_post_gate_recovery_actions,
    summarize_local_selector_readiness_rows,
    unique_sorted_u32,
)

COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_POST_GATE_DIR = (
    COMBINED_DIR / "basin_transition_post_gate_recovery_field34_cc_c2_branch_v0"
)
DEFAULT_OUTPUT_DIR = (
    COMBINED_DIR / "basin_transition_selector_source_screen_field34_cc_c2_branch_v0"
)

SOURCE_ROWS_FILENAME = "selector_source_screen_source_rows.csv"
SCORE_ROWS_FILENAME = "selector_source_screen_score_rows.csv"
READINESS_ROWS_FILENAME = "selector_source_screen_readiness_rows.csv"
SUMMARY_FILENAME = "selector_source_screen_summary.json"
CONFIG_FILENAME = "selector_source_screen_config.json"
REPORT_FILENAME = "selector_source_screen_report.md"

def _parse_csv_tuple(value: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not str(value).strip():
        return default
    return tuple(part.strip() for part in str(value).split(",") if part.strip())

def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}

def _source_case_token(value: str) -> str:
    return (
        str(value)
        .replace(":", "_")
        .replace("/", "_")
        .replace(";", "_")
        .replace("=", "_")
    )

def _context_nodes_from_recorded_path(
    recorded_path: pd.DataFrame,
    *,
    context_mode: str,
) -> np.ndarray:
    if recorded_path.empty:
        return np.asarray([], dtype=np.uint32)
    selected_rows = recorded_path[
        recorded_path.get("selected_node_ids", pd.Series("", index=recorded_path.index))
        .fillna("")
        .astype(str)
        .str.len()
        > 0
    ]
    if selected_rows.empty:
        return np.asarray([], dtype=np.uint32)
    if context_mode == "last_action":
        return _parse_node_ids(selected_rows.iloc[-1]["selected_node_ids"])
    if context_mode != "path_action_union":
        raise ValueError(f"Unsupported context mode: {context_mode}")
    nodes: list[int] = []
    for value in selected_rows["selected_node_ids"]:
        nodes.extend(int(node) for node in _parse_node_ids(value))
    return unique_sorted_u32(nodes)

def _select_source_rows(
    path_summary: pd.DataFrame,
    *,
    source_verdicts: tuple[str, ...],
    max_sources: int,
    max_sources_per_prefix: int,
) -> pd.DataFrame:
    rows = path_summary[
        path_summary["post_gate_verdict"].astype(str).isin(set(source_verdicts))
    ].copy()
    if rows.empty:
        return rows
    sort_columns = [
        "post_gate_verdict",
        "post_gate_best_delta_q_gain_from_gate",
        "post_gate_final_support",
        "post_gate_final_target_progress",
        "post_gate_step_count",
    ]
    existing_sort_columns = [column for column in sort_columns if column in rows.columns]
    rows = rows.sort_values(
        existing_sort_columns,
        ascending=[True, False, False, False, True][: len(existing_sort_columns)],
    )
    if int(max_sources_per_prefix) > 0:
        rows = (
            rows.groupby("path_prefix_rank", group_keys=False, sort=True)
            .head(int(max_sources_per_prefix))
            .copy()
        )
    if int(max_sources) > 0:
        rows = rows.head(int(max_sources)).copy()
    rows["screen_source_index"] = np.arange(1, len(rows) + 1, dtype=np.int64)
    rows["source_case"] = [
        f"p{int(row['path_prefix_rank'])}_s{int(row['screen_source_index'])}"
        for _, row in rows.iterrows()
    ]
    return rows

def _load_recorded_state_rows(post_gate_config: dict[str, Any]) -> pd.DataFrame | None:
    state_dir = post_gate_config.get("state_dir")
    state_filename = post_gate_config.get("state_rows_filename")
    if not state_dir or not state_filename:
        return None
    path = Path(str(state_dir)) / str(state_filename)
    if not path.exists():
        return None
    return pd.read_csv(path)

def _build_report(
    path: Path,
    *,
    summary: dict[str, Any],
    source_rows: pd.DataFrame,
    readiness_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Selector Source Screen",
        "",
        "This artifact screens post-gate source states before running expensive",
        "local-selector replay.  It reports whether a source has enough positive",
        "attachment-margin handles to justify follow-up.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "post_gate_dir",
        "output_dir",
        "selected_post_gate_source_count",
        "source_count",
        "score_row_count",
        "ready_count",
        "label_completion_count",
        "verdict_counts",
        "context_mode",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Source Rows", ""])
    source_cols = [
        "source_case",
        "pair_id",
        "path_prefix_rank",
        "path_policy",
        "path_selection_policy",
        "post_gate_verdict",
        "source_delta_q_vs_start",
        "source_support_distance_to_vanilla",
        "source_target_progress_from_vanilla",
        "source_context_node_count",
        "path_final_state_id",
    ]
    lines.extend(
        _markdown_table(source_rows[[c for c in source_cols if c in source_rows]], max_rows=80)
    )

    lines.extend(["", "## Readiness Rows", ""])
    readiness_cols = [
        "source_case",
        "readiness_verdict",
        "already_recovered",
        "positive_margin_node_count",
        "positive_margin_non_source_count",
        "positive_margin_candidate_label_count",
        "top_candidate_label",
        "top_label_positive_node_count",
        "top_label_node_count",
        "best_non_source_node",
        "best_non_source_margin",
        "source_delta_q_vs_start",
        "source_support_distance_to_vanilla",
    ]
    lines.extend(
        _markdown_table(
            readiness_rows[[c for c in readiness_cols if c in readiness_rows]],
            max_rows=80,
        )
    )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `selector_test_ready` and `coherent_label_completion_probe` are the only",
            "  verdicts that should normally trigger expensive local-selector replay.",
            "- `already_recovered_control` is useful as a control, but it is not a",
            "  selector validation source.",
            "- This screen uses replayed source states and the selected context mode; it",
            "  is a diagnostic filter, not an operator acceptance policy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_screen(
    *,
    post_gate_dir: Path,
    prefix_dir: Path,
    profile_batch_dir: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    source_verdicts: tuple[str, ...],
    context_mode: str,
    recovery_action_types: tuple[str, ...],
    recovery_context_multiplier: float,
    max_recovery_context_nodes: int,
    max_sources: int,
    max_sources_per_prefix: int,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    target_action_multiplier: float,
    max_target_action_nodes: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    polish_seed_offset: int,
    min_positive_margin_nodes: int,
    min_positive_margin_non_source_nodes: int,
    min_positive_margin_candidate_labels: int,
    min_source_support_distance: float,
    recovered_quality_threshold: float,
    recovered_support_threshold: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    post_gate_config = _load_json(post_gate_dir / "post_gate_recovery_config.json")
    path_summary = pd.read_csv(post_gate_dir / POST_GATE_PATH_SUMMARY_FILENAME)
    recorded_state_rows = _load_recorded_state_rows(post_gate_config)
    source_candidates = _select_source_rows(
        path_summary,
        source_verdicts=source_verdicts,
        max_sources=max_sources,
        max_sources_per_prefix=max_sources_per_prefix,
    )
    source_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    for _, source_path in source_candidates.iterrows():
        pair_id = str(source_path["pair_id"])
        prefix_rank = int(source_path["path_prefix_rank"])
        prefixes = select_prefix_rows(
            pd.read_csv(prefix_dir / BARRIER_PREFIX_ROWS_FILENAME),
            pair_ids=(pair_id,),
            top_prefixes_per_case=max(prefix_rank, 10),
        )
        prefixes = _rank_and_filter_prefix_rows(
            prefixes,
            selected_prefix_ranks=(prefix_rank,),
        )
        if prefixes.empty:
            continue
        prefix_row = prefixes.iloc[0]
        case_ctx = _load_case_context(
            prefix_row=prefix_row,
            profile_batch_dir=profile_batch_dir,
            candidate_dirs=candidate_dirs,
            vanilla_dir=vanilla_dir,
            baseline_iterations=baseline_iterations,
            candidate_polish_iterations=candidate_polish_iterations,
            resolution=resolution,
            randomness=randomness,
            perturb_seed_offset=perturb_seed_offset,
        )
        source_state, source_row, _, _ = _replay_to_source_state(
            prefix_row=prefix_row,
            source_path=source_path,
            case_ctx=case_ctx,
            target_action_multiplier=target_action_multiplier,
            max_target_action_nodes=max_target_action_nodes,
            cumulative_fraction=0.80,
            min_score_fraction=0.05,
            min_gap_fraction=0.25,
            min_guarded_pull_fraction=0.50,
            local_polish_iterations=local_polish_iterations,
            resolution=resolution,
            randomness=randomness,
            polish_seed_offset=polish_seed_offset,
            min_support_shift_from_vanilla=0.01,
            min_material_q_gain=0.01,
            recorded_state_rows=recorded_state_rows,
        )
        recorded_path = pd.DataFrame()
        if recorded_state_rows is not None:
            try:
                recorded_path = _recorded_path_to_state(
                    recorded_state_rows,
                    str(source_path["path_final_state_id"]),
                )
            except ValueError:
                recorded_path = pd.DataFrame()
        context_specs: list[dict[str, Any]] = []
        if context_mode == "recovery_contexts":
            arrays = case_ctx["arrays"]
            candidates = build_post_gate_recovery_actions(
                state=source_state,
                candidate_membership=case_ctx["candidate"].recreated.membership,
                vanilla_membership=case_ctx["vanilla"].membership,
                src=np.asarray(arrays.src, dtype=np.uint32),
                dst=np.asarray(arrays.dst, dtype=np.uint32),
                weight=np.asarray(arrays.weight, dtype=np.float64),
                node_count=int(case_ctx["baseline"].membership.size),
                action_types=recovery_action_types,
                context_multiplier=recovery_context_multiplier,
                max_context_nodes=max_recovery_context_nodes,
                include_context_only=True,
                include_candidate_transplant=False,
                include_boundary_transplant=False,
            )
            for candidate in candidates:
                context_nodes = unique_sorted_u32(candidate.action.context_nodes)
                if context_nodes.size:
                    context_specs.append(
                        {
                            "context_nodes": context_nodes,
                            "context_label": candidate.recovery_policy,
                            "context_source_action_type": candidate.source_action_type,
                            "context_move_kind": candidate.move_kind,
                        }
                    )
        else:
            context_nodes = _context_nodes_from_recorded_path(
                recorded_path,
                context_mode=context_mode,
            )
            if context_nodes.size == 0:
                context_nodes = unique_sorted_u32(
                    getattr(source_state, "action_nodes", [])
                )
            context_specs.append(
                {
                    "context_nodes": context_nodes,
                    "context_label": context_mode,
                    "context_source_action_type": "",
                    "context_move_kind": "path_context",
                }
            )
        for context_index, spec in enumerate(context_specs, start=1):
            context_nodes = unique_sorted_u32(spec["context_nodes"])
            if context_nodes.size == 0:
                continue
            base_case = str(source_path["source_case"])
            source_case = (
                base_case
                if len(context_specs) == 1 and context_mode != "recovery_contexts"
                else f"{base_case}_{context_index}_{_source_case_token(spec['context_label'])}"
            )
            score_rows = _candidate_rows(
                source_state=source_state,
                gate_nodes=context_nodes,
                full_context_nodes=context_nodes,
                moved_trace_nodes=np.asarray([], dtype=np.uint32),
                case_ctx=case_ctx,
            )
            score_rows.insert(0, "source_case", source_case)
            score_rows.insert(1, "prefix_rank", prefix_rank)
            score_rows.insert(
                2,
                "screen_source_index",
                int(source_path["screen_source_index"]),
            )
            score_rows.insert(3, "source_context_label", spec["context_label"])
            score_frames.append(score_rows)
            source_rows.append(
                {
                    "source_case": source_case,
                    "source_base_case": base_case,
                    "pair_id": pair_id,
                    "path_prefix_rank": prefix_rank,
                    "path_policy": source_path.get("path_policy", ""),
                    "path_selection_policy": source_path.get(
                        "path_selection_policy",
                        "",
                    ),
                    "post_gate_verdict": source_path.get("post_gate_verdict", ""),
                    "source_delta_q_vs_start": float(
                        source_row["state_delta_q_vs_start"]
                    ),
                    "source_support_distance_to_vanilla": float(
                        source_row["state_support_distance_to_vanilla"]
                    ),
                    "source_target_progress_from_vanilla": float(
                        source_row["state_target_progress_from_vanilla"]
                    ),
                    "source_context_node_count": int(context_nodes.size),
                    "source_context_mode": context_mode,
                    "source_context_label": spec["context_label"],
                    "source_context_source_action_type": spec[
                        "context_source_action_type"
                    ],
                    "source_context_move_kind": spec["context_move_kind"],
                    "source_context_node_ids": ",".join(
                        str(int(n)) for n in context_nodes
                    ),
                    "path_final_state_id": source_path.get("path_final_state_id", ""),
                    "prefix_context": json.dumps(
                        _prefix_context(prefix_row),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

    source_frame = pd.DataFrame(source_rows)
    score_frame = (
        pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    )
    readiness_rows = summarize_local_selector_readiness_rows(
        score_frame,
        source_summary_rows=source_frame.rename(
            columns={"path_prefix_rank": "prefix_rank"}
        ),
        min_positive_margin_nodes=min_positive_margin_nodes,
        min_positive_margin_non_source_nodes=min_positive_margin_non_source_nodes,
        min_positive_margin_candidate_labels=min_positive_margin_candidate_labels,
        min_source_support_distance=min_source_support_distance,
        recovered_quality_threshold=recovered_quality_threshold,
        recovered_support_threshold=recovered_support_threshold,
    )

    source_frame.to_csv(output_dir / SOURCE_ROWS_FILENAME, index=False)
    score_frame.to_csv(output_dir / SCORE_ROWS_FILENAME, index=False)
    readiness_rows.to_csv(output_dir / READINESS_ROWS_FILENAME, index=False)
    verdict_counts = (
        readiness_rows["readiness_verdict"].astype(str).value_counts().to_dict()
        if not readiness_rows.empty
        else {}
    )
    summary = {
        "schema": "leiden_basin_selector_source_screen.v0",
        "post_gate_dir": str(post_gate_dir),
        "output_dir": str(output_dir),
        "source_verdicts": list(source_verdicts),
        "context_mode": context_mode,
        "selected_post_gate_source_count": int(len(source_candidates)),
        "source_count": int(len(source_frame)),
        "score_row_count": int(len(score_frame)),
        "ready_count": int(verdict_counts.get("selector_test_ready", 0)),
        "label_completion_count": int(
            verdict_counts.get("coherent_label_completion_probe", 0)
        ),
        "verdict_counts": verdict_counts,
    }
    config = {
        "post_gate_dir": str(post_gate_dir),
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "source_verdicts": list(source_verdicts),
        "context_mode": context_mode,
        "recovery_action_types": list(recovery_action_types),
        "recovery_context_multiplier": float(recovery_context_multiplier),
        "max_recovery_context_nodes": int(max_recovery_context_nodes),
        "max_sources": int(max_sources),
        "max_sources_per_prefix": int(max_sources_per_prefix),
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "target_action_multiplier": float(target_action_multiplier),
        "max_target_action_nodes": int(max_target_action_nodes),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "perturb_seed_offset": int(perturb_seed_offset),
        "polish_seed_offset": int(polish_seed_offset),
        "min_positive_margin_nodes": int(min_positive_margin_nodes),
        "min_positive_margin_non_source_nodes": int(
            min_positive_margin_non_source_nodes
        ),
        "min_positive_margin_candidate_labels": int(
            min_positive_margin_candidate_labels
        ),
        "min_source_support_distance": float(min_source_support_distance),
        "recovered_quality_threshold": float(recovered_quality_threshold),
        "recovered_support_threshold": float(recovered_support_threshold),
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _build_report(
        output_dir / REPORT_FILENAME,
        summary=summary,
        source_rows=source_frame,
        readiness_rows=readiness_rows,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-gate-dir", type=Path, default=DEFAULT_POST_GATE_DIR)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--profile-batch-dir", type=Path, default=DEFAULT_PROFILE_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument(
        "--source-verdicts",
        default=",".join(
            (
                POST_GATE_VERDICT_NEAR_MISS,
                POST_GATE_VERDICT_SUPPORT_TRADEOFF,
                POST_GATE_VERDICT_PLATEAU,
            )
        ),
    )
    parser.add_argument(
        "--context-mode",
        choices=("path_action_union", "last_action", "recovery_contexts"),
        default="path_action_union",
    )
    parser.add_argument(
        "--recovery-action-types",
        default=",".join(
            (
                ACTION_CANDIDATE_CLOSURE_TOPK,
                ACTION_VANILLA_CLOSURE_TOPK,
                ACTION_BOUNDARY_SHELL_TOPK,
            )
        ),
    )
    parser.add_argument("--recovery-context-multiplier", type=float, default=0.5)
    parser.add_argument("--max-recovery-context-nodes", type=int, default=64)
    parser.add_argument("--max-sources", type=int, default=20)
    parser.add_argument("--max-sources-per-prefix", type=int, default=4)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--candidate-polish-iterations", type=int, default=5)
    parser.add_argument("--local-polish-iterations", type=int, default=3)
    parser.add_argument("--target-action-multiplier", type=float, default=0.5)
    parser.add_argument("--max-target-action-nodes", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--polish-seed-offset", type=int, default=11000)
    parser.add_argument("--min-positive-margin-nodes", type=int, default=2)
    parser.add_argument("--min-positive-margin-non-source-nodes", type=int, default=2)
    parser.add_argument("--min-positive-margin-candidate-labels", type=int, default=2)
    parser.add_argument("--min-source-support-distance", type=float, default=0.01)
    parser.add_argument("--recovered-quality-threshold", type=float, default=0.01)
    parser.add_argument("--recovered-support-threshold", type=float, default=0.05)
    return parser

def main() -> int:
    args = build_parser().parse_args()
    candidate_dirs = (
        tuple(args.candidate_dir)
        if args.candidate_dir
        else tuple(DEFAULT_CANDIDATE_DIRS)
    )
    summary = run_screen(
        post_gate_dir=args.post_gate_dir,
        prefix_dir=args.prefix_dir,
        profile_batch_dir=args.profile_batch_dir,
        output_dir=args.output_dir,
        candidate_dirs=candidate_dirs,
        vanilla_dir=args.vanilla_dir,
        source_verdicts=_parse_csv_tuple(args.source_verdicts),
        context_mode=args.context_mode,
        recovery_action_types=_parse_csv_tuple(args.recovery_action_types),
        recovery_context_multiplier=args.recovery_context_multiplier,
        max_recovery_context_nodes=args.max_recovery_context_nodes,
        max_sources=args.max_sources,
        max_sources_per_prefix=args.max_sources_per_prefix,
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        target_action_multiplier=args.target_action_multiplier,
        max_target_action_nodes=args.max_target_action_nodes,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        polish_seed_offset=args.polish_seed_offset,
        min_positive_margin_nodes=args.min_positive_margin_nodes,
        min_positive_margin_non_source_nodes=args.min_positive_margin_non_source_nodes,
        min_positive_margin_candidate_labels=args.min_positive_margin_candidate_labels,
        min_source_support_distance=args.min_source_support_distance,
        recovered_quality_threshold=args.recovered_quality_threshold,
        recovered_support_threshold=args.recovered_support_threshold,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
