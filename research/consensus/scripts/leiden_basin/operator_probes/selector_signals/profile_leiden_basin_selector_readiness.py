#!/usr/bin/env python3
"""Profile which source-local selector artifacts are worth probing next."""

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
    LOCAL_SELECTOR_READINESS_LABEL_COMPLETION,
    LOCAL_SELECTOR_READINESS_READY,
    summarize_local_selector_readiness_rows,
)

COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_OUTPUT_DIR = COMBINED_DIR / "basin_transition_local_selector_readiness_field34_cc_v0"

ATTACHMENT_SCORE_ROWS_FILENAME = "attachment_margin_cross_prefix_score_rows.csv"
ATTACHMENT_SUMMARY_ROWS_FILENAME = "attachment_margin_cross_prefix_summary_rows.csv"
ROWS_FILENAME = "local_selector_readiness_rows.csv"
SUMMARY_FILENAME = "local_selector_readiness_summary.json"
CONFIG_FILENAME = "local_selector_readiness_config.json"
REPORT_FILENAME = "local_selector_readiness_report.md"

def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 60) -> list[str]:
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
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines

def _discover_attachment_dirs(root_dir: Path, *, pattern: str) -> tuple[Path, ...]:
    if not root_dir.exists():
        return ()
    dirs: list[Path] = []
    for path in sorted(root_dir.glob(pattern)):
        if not path.is_dir():
            continue
        if (path / ATTACHMENT_SCORE_ROWS_FILENAME).exists():
            dirs.append(path)
    return tuple(dirs)

def _load_summary_rows(directory: Path) -> pd.DataFrame:
    path = directory / ATTACHMENT_SUMMARY_ROWS_FILENAME
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def _write_report(path: Path, *, rows: pd.DataFrame, summary: dict[str, Any]) -> None:
    display_cols = [
        "attachment_artifact",
        "source_case",
        "prefix_rank",
        "source_recovery_index",
        "readiness_verdict",
        "source_delta_q_vs_start",
        "source_support_distance_to_vanilla",
        "source_target_progress_from_vanilla",
        "positive_margin_non_source_count",
        "positive_margin_candidate_label_count",
        "top_candidate_label",
        "top_label_positive_margin_sum",
        "top_label_positive_node_count",
        "top_label_node_count",
        "second_label_positive_margin_sum",
        "label_competition_gap",
        "best_non_source_node",
        "best_non_source_margin",
    ]
    ready = rows[
        rows["readiness_verdict"].astype(str).eq(LOCAL_SELECTOR_READINESS_READY)
    ].copy()
    if not ready.empty:
        ready = ready.sort_values(
            [
                "positive_margin_candidate_label_count",
                "positive_margin_non_source_count",
                "source_delta_q_vs_start",
            ],
            ascending=[False, False, True],
        )
    verdict_counts = pd.DataFrame(
        [
            {"readiness_verdict": key, "row_count": value}
            for key, value in summary.get("verdict_counts", {}).items()
        ]
    )
    lines = [
        "# Local Selector Readiness Profile",
        "",
        "This artifact ranks source-local attachment-margin tables by whether they",
        "can meaningfully test a local handle selector.  It is a diagnostic inventory,",
        "not an operator result.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "root_dir",
        "output_dir",
        "attachment_dir_count",
        "source_case_count",
        "ready_count",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Verdict Counts", ""])
    lines.extend(_markdown_table(verdict_counts, max_rows=20))
    lines.extend(["", "## Ready Rows", ""])
    lines.extend(
        _markdown_table(ready[[c for c in display_cols if c in ready]], max_rows=80)
    )
    lines.extend(["", "## All Rows", ""])
    ordered = rows.sort_values(["readiness_verdict", "attachment_artifact", "source_case"])
    lines.extend(
        _markdown_table(
            ordered[[c for c in display_cols if c in ordered]],
            max_rows=160,
        )
    )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `selector_test_ready` means there are multiple non-source positive-margin",
            "  handles across multiple candidate labels and the source is not already",
            "  QF/support recovered.",
            f"- `{LOCAL_SELECTOR_READINESS_LABEL_COMPLETION}` means there may be only one",
            "  positive anchor, but its candidate-label group has enough nodes to test",
            "  same-label completion.",
            "- `already_recovered_control` rows are useful negative controls, but weak",
            "  tests of the selector mechanism.",
            "- `too_few_handles` and `no_label_competition` should not be promoted into",
            "  selector-validation claims because there is little for the selector to choose.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_profile(
    *,
    root_dir: Path,
    attachment_dirs: tuple[Path, ...],
    output_dir: Path,
    min_positive_margin_nodes: int,
    min_positive_margin_non_source_nodes: int,
    min_positive_margin_candidate_labels: int,
    min_source_support_distance: float,
    recovered_quality_threshold: float,
    recovered_support_threshold: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    for directory in attachment_dirs:
        score_path = directory / ATTACHMENT_SCORE_ROWS_FILENAME
        if not score_path.exists():
            continue
        score_rows = pd.read_csv(score_path)
        summary_rows = _load_summary_rows(directory)
        rows = summarize_local_selector_readiness_rows(
            score_rows,
            source_summary_rows=summary_rows,
            min_positive_margin_nodes=min_positive_margin_nodes,
            min_positive_margin_non_source_nodes=min_positive_margin_non_source_nodes,
            min_positive_margin_candidate_labels=min_positive_margin_candidate_labels,
            min_source_support_distance=min_source_support_distance,
            recovered_quality_threshold=recovered_quality_threshold,
            recovered_support_threshold=recovered_support_threshold,
        )
        if rows.empty:
            continue
        rows.insert(0, "attachment_artifact", directory.name)
        rows.insert(1, "attachment_dir", str(directory))
        all_rows.append(rows)
    rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    verdict_counts = (
        rows["readiness_verdict"].astype(str).value_counts().to_dict()
        if not rows.empty
        else {}
    )
    summary = {
        "schema": "leiden_basin_local_selector_readiness.v0",
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "attachment_dirs": [str(path) for path in attachment_dirs],
        "attachment_dir_count": int(len(attachment_dirs)),
        "source_case_count": int(len(rows)),
        "ready_count": int(verdict_counts.get(LOCAL_SELECTOR_READINESS_READY, 0)),
        "label_completion_count": int(
            verdict_counts.get(LOCAL_SELECTOR_READINESS_LABEL_COMPLETION, 0)
        ),
        "verdict_counts": verdict_counts,
        "thresholds": {
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
        },
        "paths": {
            "rows": str(output_dir / ROWS_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "root_dir": str(root_dir),
        "attachment_dirs": [str(path) for path in attachment_dirs],
        "output_dir": str(output_dir),
        **summary["thresholds"],
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_FILENAME, rows=rows, summary=summary)
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=COMBINED_DIR)
    parser.add_argument("--attachment-dir", type=Path, action="append", default=None)
    parser.add_argument("--artifact-pattern", default="basin_transition_attachment_margin_cross_prefix_*")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-positive-margin-nodes", type=int, default=2)
    parser.add_argument("--min-positive-margin-non-source-nodes", type=int, default=2)
    parser.add_argument("--min-positive-margin-candidate-labels", type=int, default=2)
    parser.add_argument("--min-source-support-distance", type=float, default=0.01)
    parser.add_argument("--recovered-quality-threshold", type=float, default=0.01)
    parser.add_argument("--recovered-support-threshold", type=float, default=0.05)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    attachment_dirs = (
        tuple(args.attachment_dir)
        if args.attachment_dir
        else _discover_attachment_dirs(args.root_dir, pattern=args.artifact_pattern)
    )
    summary = run_profile(
        root_dir=args.root_dir,
        attachment_dirs=attachment_dirs,
        output_dir=args.output_dir,
        min_positive_margin_nodes=args.min_positive_margin_nodes,
        min_positive_margin_non_source_nodes=args.min_positive_margin_non_source_nodes,
        min_positive_margin_candidate_labels=args.min_positive_margin_candidate_labels,
        min_source_support_distance=args.min_source_support_distance,
        recovered_quality_threshold=args.recovered_quality_threshold,
        recovered_support_threshold=args.recovered_support_threshold,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
