#!/usr/bin/env python3
"""Batch selector-source screening over post-gate recovery artifacts."""

from __future__ import annotations

import argparse
import json
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
from pandas.errors import EmptyDataError

SCRIPT_DIR = Path(__file__).resolve().parent

from probe_leiden_basin_post_gate_recovery_moves import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_PROFILE_BATCH_DIR,
    DEFAULT_VANILLA_DIR,
    POST_GATE_PATH_SUMMARY_FILENAME,
)
from screen_leiden_basin_selector_sources import (  # noqa: E402
    COMBINED_DIR,
    DEFAULT_PREFIX_DIR,
    READINESS_ROWS_FILENAME as SINGLE_READINESS_ROWS_FILENAME,
    SOURCE_ROWS_FILENAME as SINGLE_SOURCE_ROWS_FILENAME,
    SUMMARY_FILENAME as SINGLE_SUMMARY_FILENAME,
    _markdown_table,
    _parse_csv_tuple,
    run_screen,
)
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    ACTION_BOUNDARY_SHELL_TOPK,
    ACTION_CANDIDATE_CLOSURE_TOPK,
    ACTION_VANILLA_CLOSURE_TOPK,
    LOCAL_SELECTOR_READINESS_LABEL_COMPLETION,
    LOCAL_SELECTOR_READINESS_READY,
    POST_GATE_VERDICT_NEAR_MISS,
    POST_GATE_VERDICT_PLATEAU,
    POST_GATE_VERDICT_SUPPORT_TRADEOFF,
)

DEFAULT_OUTPUT_DIR = (
    COMBINED_DIR / "basin_transition_selector_source_screen_batch_field34_cc_non_c0_v0"
)

BATCH_ROWS_FILENAME = "selector_source_screen_batch_rows.csv"
BATCH_SOURCE_ROWS_FILENAME = "selector_source_screen_batch_source_rows.csv"
BATCH_READINESS_ROWS_FILENAME = "selector_source_screen_batch_readiness_rows.csv"
SUMMARY_FILENAME = "selector_source_screen_batch_summary.json"
CONFIG_FILENAME = "selector_source_screen_batch_config.json"
REPORT_FILENAME = "selector_source_screen_batch_report.md"

def _discover_post_gate_dirs(root_dir: Path, *, pattern: str) -> tuple[Path, ...]:
    if not root_dir.exists():
        return ()
    dirs: list[Path] = []
    for path in sorted(root_dir.glob(pattern)):
        if not path.is_dir():
            continue
        if (path / POST_GATE_PATH_SUMMARY_FILENAME).exists():
            dirs.append(path)
    return tuple(dirs)

def _pair_ids_for_post_gate_dir(post_gate_dir: Path) -> tuple[str, ...]:
    path = post_gate_dir / POST_GATE_PATH_SUMMARY_FILENAME
    if not path.exists():
        return ()
    rows = pd.read_csv(path, usecols=lambda column: column == "pair_id")
    if "pair_id" not in rows:
        return ()
    return tuple(sorted(rows["pair_id"].dropna().astype(str).unique()))

def _pair_prefix_allowed(
    pair_ids: tuple[str, ...],
    *,
    include_pair_prefixes: tuple[str, ...],
    exclude_pair_prefixes: tuple[str, ...],
) -> bool:
    if include_pair_prefixes and not any(
        pair_id.startswith(prefix)
        for pair_id in pair_ids
        for prefix in include_pair_prefixes
    ):
        return False
    if exclude_pair_prefixes and any(
        pair_id.startswith(prefix)
        for pair_id in pair_ids
        for prefix in exclude_pair_prefixes
    ):
        return False
    return True

def _select_post_gate_dirs(
    post_gate_dirs: tuple[Path, ...],
    *,
    include_pair_prefixes: tuple[str, ...],
    exclude_pair_prefixes: tuple[str, ...],
    max_artifacts: int,
) -> tuple[Path, ...]:
    selected: list[Path] = []
    for post_gate_dir in post_gate_dirs:
        pair_ids = _pair_ids_for_post_gate_dir(post_gate_dir)
        if not pair_ids:
            continue
        if not _pair_prefix_allowed(
            pair_ids,
            include_pair_prefixes=include_pair_prefixes,
            exclude_pair_prefixes=exclude_pair_prefixes,
        ):
            continue
        selected.append(post_gate_dir)
        if int(max_artifacts) > 0 and len(selected) >= int(max_artifacts):
            break
    return tuple(selected)

def _screen_output_dir(output_dir: Path, post_gate_dir: Path) -> Path:
    name = post_gate_dir.name.replace(
        "basin_transition_post_gate_recovery_",
        "selector_source_screen_",
        1,
    )
    return output_dir / name

def _write_report(
    path: Path,
    *,
    batch_rows: pd.DataFrame,
    readiness_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    ready_verdicts = {
        LOCAL_SELECTOR_READINESS_READY,
        LOCAL_SELECTOR_READINESS_LABEL_COMPLETION,
    }
    replay_candidates = (
        readiness_rows[
            readiness_rows["readiness_verdict"].astype(str).isin(ready_verdicts)
        ].copy()
        if not readiness_rows.empty
        else pd.DataFrame()
    )
    if not replay_candidates.empty:
        replay_candidates = replay_candidates.sort_values(
            [
                "already_recovered",
                "positive_margin_candidate_label_count",
                "positive_margin_non_source_count",
                "source_delta_q_vs_start",
            ],
            ascending=[True, False, False, True],
        )
    lines = [
        "# Selector Source Screen Batch",
        "",
        "This artifact applies the selector-source screen across post-gate recovery",
        "artifacts.  It is a budget gate for later local-selector replay, not an",
        "operator result.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "root_dir",
        "output_dir",
        "post_gate_dir_count",
        "screened_artifact_count",
        "selected_post_gate_source_count",
        "source_context_variant_count",
        "ready_count",
        "label_completion_count",
        "verdict_counts",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Artifact Rows", ""])
    artifact_cols = [
        "post_gate_artifact",
        "pair_ids",
        "context_mode",
        "selected_post_gate_source_count",
        "source_count",
        "score_row_count",
        "ready_count",
        "label_completion_count",
        "verdict_counts",
        "screen_output_dir",
    ]
    lines.extend(
        _markdown_table(
            batch_rows[[c for c in artifact_cols if c in batch_rows]],
            max_rows=80,
        )
    )

    lines.extend(["", "## Replay Candidates", ""])
    readiness_cols = [
        "post_gate_artifact",
        "source_case",
        "readiness_verdict",
        "already_recovered",
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
            replay_candidates[[c for c in readiness_cols if c in replay_candidates]],
            max_rows=120,
        )
    )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Replay budget should normally go only to `selector_test_ready` or",
            "  `coherent_label_completion_probe` rows that are not already recovered.",
            "- `already_recovered_control` rows are useful controls, but weak selector",
            "  validation targets.",
            "- A batch with zero replay candidates means the next step should generate a",
            "  fresh post-gate source slice instead of rerunning selector replay.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _load_child_rows(
    screen_dir: Path,
    *,
    filename: str,
    post_gate_artifact: str,
    post_gate_dir: Path,
) -> pd.DataFrame:
    path = screen_dir / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        rows = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    if rows.empty:
        return rows
    rows.insert(0, "post_gate_artifact", post_gate_artifact)
    rows.insert(1, "post_gate_dir", str(post_gate_dir))
    rows.insert(2, "screen_output_dir", str(screen_dir))
    return rows

def run_batch(
    *,
    root_dir: Path,
    post_gate_dirs: tuple[Path, ...],
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
    batch_rows: list[dict[str, Any]] = []
    source_frames: list[pd.DataFrame] = []
    readiness_frames: list[pd.DataFrame] = []
    for post_gate_dir in post_gate_dirs:
        screen_dir = _screen_output_dir(output_dir, post_gate_dir)
        summary = run_screen(
            post_gate_dir=post_gate_dir,
            prefix_dir=prefix_dir,
            profile_batch_dir=profile_batch_dir,
            output_dir=screen_dir,
            candidate_dirs=candidate_dirs,
            vanilla_dir=vanilla_dir,
            source_verdicts=source_verdicts,
            context_mode=context_mode,
            recovery_action_types=recovery_action_types,
            recovery_context_multiplier=recovery_context_multiplier,
            max_recovery_context_nodes=max_recovery_context_nodes,
            max_sources=max_sources,
            max_sources_per_prefix=max_sources_per_prefix,
            baseline_iterations=baseline_iterations,
            candidate_polish_iterations=candidate_polish_iterations,
            local_polish_iterations=local_polish_iterations,
            target_action_multiplier=target_action_multiplier,
            max_target_action_nodes=max_target_action_nodes,
            resolution=resolution,
            randomness=randomness,
            perturb_seed_offset=perturb_seed_offset,
            polish_seed_offset=polish_seed_offset,
            min_positive_margin_nodes=min_positive_margin_nodes,
            min_positive_margin_non_source_nodes=min_positive_margin_non_source_nodes,
            min_positive_margin_candidate_labels=min_positive_margin_candidate_labels,
            min_source_support_distance=min_source_support_distance,
            recovered_quality_threshold=recovered_quality_threshold,
            recovered_support_threshold=recovered_support_threshold,
        )
        summary_path = screen_dir / SINGLE_SUMMARY_FILENAME
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pair_ids = _pair_ids_for_post_gate_dir(post_gate_dir)
        batch_rows.append(
            {
                "post_gate_artifact": post_gate_dir.name,
                "post_gate_dir": str(post_gate_dir),
                "pair_ids": ",".join(pair_ids),
                "screen_output_dir": str(screen_dir),
                "context_mode": summary.get("context_mode", context_mode),
                "selected_post_gate_source_count": int(
                    summary.get("selected_post_gate_source_count", 0)
                ),
                "source_count": int(summary.get("source_count", 0)),
                "score_row_count": int(summary.get("score_row_count", 0)),
                "ready_count": int(summary.get("ready_count", 0)),
                "label_completion_count": int(
                    summary.get("label_completion_count", 0)
                ),
                "verdict_counts": json.dumps(
                    summary.get("verdict_counts", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        source_frames.append(
            _load_child_rows(
                screen_dir,
                filename=SINGLE_SOURCE_ROWS_FILENAME,
                post_gate_artifact=post_gate_dir.name,
                post_gate_dir=post_gate_dir,
            )
        )
        readiness_frames.append(
            _load_child_rows(
                screen_dir,
                filename=SINGLE_READINESS_ROWS_FILENAME,
                post_gate_artifact=post_gate_dir.name,
                post_gate_dir=post_gate_dir,
            )
        )

    batch_frame = pd.DataFrame(batch_rows)
    source_frame = (
        pd.concat([frame for frame in source_frames if not frame.empty], ignore_index=True)
        if source_frames
        else pd.DataFrame()
    )
    readiness_frame = (
        pd.concat(
            [frame for frame in readiness_frames if not frame.empty],
            ignore_index=True,
        )
        if readiness_frames
        else pd.DataFrame()
    )
    batch_frame.to_csv(output_dir / BATCH_ROWS_FILENAME, index=False)
    source_frame.to_csv(output_dir / BATCH_SOURCE_ROWS_FILENAME, index=False)
    readiness_frame.to_csv(output_dir / BATCH_READINESS_ROWS_FILENAME, index=False)
    verdict_counts = (
        readiness_frame["readiness_verdict"].astype(str).value_counts().to_dict()
        if not readiness_frame.empty
        else {}
    )
    summary = {
        "schema": "leiden_basin_selector_source_screen_batch.v0",
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "post_gate_dirs": [str(path) for path in post_gate_dirs],
        "post_gate_dir_count": int(len(post_gate_dirs)),
        "screened_artifact_count": int(len(batch_frame)),
        "selected_post_gate_source_count": int(
            batch_frame["selected_post_gate_source_count"].sum()
            if not batch_frame.empty
            else 0
        ),
        "source_context_variant_count": int(
            batch_frame["source_count"].sum() if not batch_frame.empty else 0
        ),
        "score_row_count": int(
            batch_frame["score_row_count"].sum() if not batch_frame.empty else 0
        ),
        "ready_count": int(verdict_counts.get(LOCAL_SELECTOR_READINESS_READY, 0)),
        "label_completion_count": int(
            verdict_counts.get(LOCAL_SELECTOR_READINESS_LABEL_COMPLETION, 0)
        ),
        "verdict_counts": verdict_counts,
        "paths": {
            "batch_rows": str(output_dir / BATCH_ROWS_FILENAME),
            "source_rows": str(output_dir / BATCH_SOURCE_ROWS_FILENAME),
            "readiness_rows": str(output_dir / BATCH_READINESS_ROWS_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "root_dir": str(root_dir),
        "post_gate_dirs": [str(path) for path in post_gate_dirs],
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "output_dir": str(output_dir),
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
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / REPORT_FILENAME,
        batch_rows=batch_frame,
        readiness_rows=readiness_frame,
        summary=summary,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=COMBINED_DIR)
    parser.add_argument("--post-gate-dir", type=Path, action="append", default=None)
    parser.add_argument(
        "--artifact-pattern",
        default="basin_transition_post_gate_recovery_field34_cc_*",
    )
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--profile-batch-dir", type=Path, default=DEFAULT_PROFILE_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--include-pair-prefixes", default="")
    parser.add_argument("--exclude-pair-prefixes", default="c0")
    parser.add_argument("--max-artifacts", type=int, default=0)
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
        default="recovery_contexts",
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
    parser.add_argument("--max-sources", type=int, default=5)
    parser.add_argument("--max-sources-per-prefix", type=int, default=2)
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
    discovered = (
        tuple(args.post_gate_dir)
        if args.post_gate_dir
        else _discover_post_gate_dirs(args.root_dir, pattern=args.artifact_pattern)
    )
    post_gate_dirs = _select_post_gate_dirs(
        discovered,
        include_pair_prefixes=_parse_csv_tuple(args.include_pair_prefixes),
        exclude_pair_prefixes=_parse_csv_tuple(args.exclude_pair_prefixes),
        max_artifacts=args.max_artifacts,
    )
    candidate_dirs = (
        tuple(args.candidate_dir)
        if args.candidate_dir
        else tuple(DEFAULT_CANDIDATE_DIRS)
    )
    summary = run_batch(
        root_dir=args.root_dir,
        post_gate_dirs=post_gate_dirs,
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
