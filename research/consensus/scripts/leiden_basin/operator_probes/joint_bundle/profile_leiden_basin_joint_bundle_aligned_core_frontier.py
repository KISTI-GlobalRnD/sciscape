#!/usr/bin/env python3
"""Replay promising joint-bundle rows and aggregate their aligned cores.

This is a diagnostic frontier builder for the next Dongdaemun operator. It
does not introduce a new mutation policy. It reuses the focused joint-bundle
replay path, then ranks the small label-invariant core and nearby context that
survived polish.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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

from explain_leiden_basin_attachment_margin_joint_bundle_replay import (  # noqa: E402
    DEFAULT_ATTACHMENT_DIR,
    DEFAULT_JOINT_BUNDLE_DIR,
    NODE_ROWS_FILENAME,
    SUMMARY_ROWS_FILENAME,
    run_replay,
)
from run_leiden_basin_attachment_margin_cross_prefix_probe import (  # noqa: E402
    DEFAULT_SOURCE_RECOVERY_POLICY,
)

COMBINED_DIR = DEFAULT_ATTACHMENT_DIR.parent
DEFAULT_OPERATOR_REVIEW_DIR = (
    COMBINED_DIR / "basin_evaluation_metric_audit_v0/recomputed_operator_metric_review"
)
DEFAULT_TOP_ROWS = DEFAULT_OPERATOR_REVIEW_DIR / "recomputed_operator_metric_top_rows.csv"
DEFAULT_OUTPUT_DIR = COMBINED_DIR / "joint_bundle_aligned_core_frontier_v0"

CONFIG_ROWS_FILENAME = "joint_bundle_aligned_core_replay_config_rows.csv"
SUMMARY_ROWS_FILENAME_OUT = "joint_bundle_aligned_core_replay_summary_rows.csv"
NODE_ROWS_FILENAME_OUT = "joint_bundle_aligned_core_node_rows.csv"
NODE_FRONTIER_FILENAME = "joint_bundle_aligned_core_node_frontier_rows.csv"
SUMMARY_FILENAME = "joint_bundle_aligned_core_frontier_summary.json"
REPORT_FILENAME = "joint_bundle_aligned_core_frontier_report.md"

def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "row"

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
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines

def _set_or_insert(frame: pd.DataFrame, index: int, column: str, value: Any) -> None:
    if column in frame.columns:
        frame[column] = value
    else:
        frame.insert(index, column, value)

def select_replay_configs(
    top_rows: pd.DataFrame,
    *,
    max_configs: int = 6,
    min_quality_gain: float = 0.0,
) -> pd.DataFrame:
    """Select unique positive joint-bundle configs from the recomputed review."""
    if top_rows.empty:
        return pd.DataFrame()
    required = {
        "artifact",
        "source_case",
        "target_k",
        "context_family",
        "context_multiplier",
        "move_kind",
        "quality_gain",
        "final_aligned_changed",
    }
    missing = required - set(top_rows.columns)
    if missing:
        raise ValueError(f"top rows are missing required columns: {sorted(missing)}")
    rows = top_rows[top_rows["artifact"].astype(str).eq("joint_bundle")].copy()
    rows = rows[pd.to_numeric(rows["quality_gain"], errors="coerce") > float(min_quality_gain)]
    rows = rows[
        rows["source_case"].fillna("").astype(str).str.len().gt(0)
        & rows["context_family"].fillna("").astype(str).str.len().gt(0)
        & rows["move_kind"].fillna("").astype(str).str.len().gt(0)
    ].copy()
    if rows.empty:
        return rows
    rows["_quality_gain"] = pd.to_numeric(rows["quality_gain"], errors="coerce")
    rows["_final_aligned_changed"] = pd.to_numeric(
        rows["final_aligned_changed"], errors="coerce"
    )
    rows = rows.sort_values(
        ["_quality_gain", "_final_aligned_changed", "rank"],
        ascending=[False, False, True],
        na_position="last",
    )
    dedupe_cols = [
        "source_case",
        "target_k",
        "context_family",
        "context_multiplier",
        "move_kind",
    ]
    rows = rows.drop_duplicates(dedupe_cols, keep="first").head(int(max_configs)).copy()
    out = rows[
        [
            "source_case",
            "target_k",
            "context_family",
            "context_multiplier",
            "move_kind",
            "quality_gain",
            "final_aligned_changed",
            "final_exact_changed",
            "final_exact_only_changed",
            "endpoint_distance",
            "state_delta_q_vs_vanilla",
            "joint_verdict",
        ]
    ].copy()
    out.insert(0, "config_rank", range(1, len(out) + 1))
    out["replay_slug"] = [
        _slug(
            f"{row.source_case}_tk{int(float(row.target_k))}_"
            f"{row.context_family}_cm{float(row.context_multiplier):g}_"
            f"{row.move_kind}"
        )
        for row in out.itertuples(index=False)
    ]
    return out.reset_index(drop=True)

def aggregate_node_frontier(node_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-node replay rows into a small aligned-core frontier."""
    if node_rows.empty:
        return pd.DataFrame()
    rows = node_rows.copy()
    if "source_case" not in rows:
        rows["source_case"] = ""
    for column in (
        "aligned_partition_changed",
        "exact_label_changed",
        "in_selected_target",
        "in_context",
        "in_bundle",
        "in_source_action",
        "in_source_mutable",
    ):
        if column in rows:
            rows[column] = rows[column].map(bool)
        else:
            rows[column] = False
    agg = (
        rows.groupby("node", dropna=False)
        .agg(
            replay_row_count=("replay_slug", "nunique"),
            aligned_change_count=("aligned_partition_changed", "sum"),
            exact_change_count=("exact_label_changed", "sum"),
            selected_target_count=("in_selected_target", "sum"),
            context_count=("in_context", "sum"),
            bundle_count=("in_bundle", "sum"),
            source_action_count=("in_source_action", "sum"),
            source_mutable_count=("in_source_mutable", "sum"),
            min_hop_to_target=("hop_to_selected_target", "min"),
            min_hop_to_bundle=("hop_to_bundle", "min"),
            max_pull_to_target=("pull_to_selected_target", "max"),
            max_pull_to_context=("pull_to_context", "max"),
            max_pull_to_bundle=("pull_to_bundle", "max"),
            baseline_label=("baseline_label", "first"),
            vanilla_label=("vanilla_label", "first"),
            candidate_label=("candidate_label", "first"),
            source_cases=("source_case", lambda value: ",".join(sorted(set(map(str, value))))),
            config_ranks=("config_rank", lambda value: ",".join(str(int(v)) for v in sorted(set(value)))),
        )
        .reset_index()
    )
    agg["aligned_change_fraction"] = agg["aligned_change_count"] / agg[
        "replay_row_count"
    ].clip(lower=1)
    agg["frontier_role"] = [
        "target_core"
        if target > 0 and changed > 0
        else "context_core"
        if context > 0 and changed > 0
        else "source_mutable_core"
        if source_mutable > 0 and changed > 0
        else "bundle_context"
        if bundle > 0
        else "observer"
        for target, context, source_mutable, bundle, changed in zip(
            agg["selected_target_count"],
            agg["context_count"],
            agg["source_mutable_count"],
            agg["bundle_count"],
            agg["aligned_change_count"],
            strict=False,
        )
    ]
    return agg.sort_values(
        [
            "aligned_change_count",
            "selected_target_count",
            "context_count",
            "max_pull_to_bundle",
            "node",
        ],
        ascending=[False, False, False, False, True],
    )

def _write_report(
    path: Path,
    *,
    configs: pd.DataFrame,
    summaries: pd.DataFrame,
    frontier: pd.DataFrame,
) -> None:
    config_cols = [
        "config_rank",
        "source_case",
        "target_k",
        "context_family",
        "context_multiplier",
        "move_kind",
        "quality_gain",
        "final_aligned_changed",
        "state_delta_q_vs_vanilla",
        "joint_verdict",
    ]
    summary_cols = [
        "config_rank",
        "source_case",
        "target_k",
        "context_family",
        "context_multiplier",
        "move_kind",
        "delta_q_gain_vs_source",
        "delta_q_vs_vanilla",
        "final_aligned_changed_vs_source",
        "final_exact_only_changed_vs_source",
        "endpoint_distance_to_source",
        "aligned_changed_node_ids_vs_source",
    ]
    frontier_cols = [
        "node",
        "frontier_role",
        "aligned_change_count",
        "aligned_change_fraction",
        "selected_target_count",
        "context_count",
        "source_mutable_count",
        "min_hop_to_target",
        "max_pull_to_bundle",
        "baseline_label",
        "vanilla_label",
        "candidate_label",
        "config_ranks",
    ]
    lines = [
        "# Joint-Bundle Aligned Core Frontier",
        "",
        "This diagnostic replays the strongest positive joint-bundle rows and",
        "aggregates the label-invariant aligned core that survived polish.",
        "",
        "## Selected Replay Configs",
        "",
    ]
    lines.extend(_markdown_table(configs[[c for c in config_cols if c in configs]], max_rows=20))
    lines.extend(["", "## Replay Outcomes", ""])
    lines.extend(
        _markdown_table(summaries[[c for c in summary_cols if c in summaries]], max_rows=30)
    )
    lines.extend(["", "## Aligned Core Frontier", ""])
    lines.extend(
        _markdown_table(frontier[[c for c in frontier_cols if c in frontier]], max_rows=80)
    )
    lines.extend(
        [
            "",
            "## Operator Implication",
            "",
            "- Nodes that repeatedly appear as `target_core` are direct tunneling handles.",
            "- Nodes that appear as `context_core` or `source_mutable_core` are the boundary context to price explicitly.",
            "- Large exact-only changes should remain out of the operator objective.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_frontier(
    *,
    top_rows_path: Path,
    attachment_dir: Path,
    joint_bundle_dir: Path,
    output_dir: Path,
    source_recovery_policy: str,
    recovery_seed: int,
    max_configs: int,
    min_quality_gain: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    top_rows = pd.read_csv(top_rows_path)
    configs = select_replay_configs(
        top_rows,
        max_configs=max_configs,
        min_quality_gain=min_quality_gain,
    )
    if configs.empty:
        raise ValueError("No joint-bundle replay configs selected")
    replay_root = output_dir / "replays"
    replay_root.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    node_frames: list[pd.DataFrame] = []
    for _, config in configs.iterrows():
        replay_dir = replay_root / str(config["replay_slug"])
        if not (
            (replay_dir / SUMMARY_ROWS_FILENAME).exists()
            and (replay_dir / NODE_ROWS_FILENAME).exists()
        ):
            run_replay(
                attachment_dir=attachment_dir,
                joint_bundle_dir=joint_bundle_dir,
                output_dir=replay_dir,
                source_case=str(config["source_case"]),
                target_k=int(float(config["target_k"])),
                context_family=str(config["context_family"]),
                context_multiplier=float(config["context_multiplier"]),
                move_kinds=(str(config["move_kind"]),),
                source_recovery_policy=source_recovery_policy,
                recovery_seed=recovery_seed,
            )
        summary = pd.read_csv(replay_dir / SUMMARY_ROWS_FILENAME)
        nodes = pd.read_csv(replay_dir / NODE_ROWS_FILENAME)
        for frame in (summary, nodes):
            _set_or_insert(frame, 0, "config_rank", int(config["config_rank"]))
            _set_or_insert(frame, 1, "replay_slug", str(config["replay_slug"]))
            _set_or_insert(frame, 2, "source_case", str(config["source_case"]))
            _set_or_insert(frame, 3, "target_k", int(float(config["target_k"])))
            _set_or_insert(frame, 4, "context_family", str(config["context_family"]))
            _set_or_insert(
                frame,
                5,
                "context_multiplier",
                float(config["context_multiplier"]),
            )
            _set_or_insert(frame, 6, "configured_move_kind", str(config["move_kind"]))
        summary_frames.append(summary)
        node_frames.append(nodes)

    summaries = pd.concat(summary_frames, ignore_index=True)
    node_rows = pd.concat(node_frames, ignore_index=True)
    frontier = aggregate_node_frontier(node_rows)

    configs.to_csv(output_dir / CONFIG_ROWS_FILENAME, index=False)
    summaries.to_csv(output_dir / SUMMARY_ROWS_FILENAME_OUT, index=False)
    node_rows.to_csv(output_dir / NODE_ROWS_FILENAME_OUT, index=False)
    frontier.to_csv(output_dir / NODE_FRONTIER_FILENAME, index=False)
    _write_report(output_dir / REPORT_FILENAME, configs=configs, summaries=summaries, frontier=frontier)

    payload = {
        "schema": "leiden_basin_joint_bundle_aligned_core_frontier.v0",
        "output_dir": str(output_dir),
        "config_count": int(len(configs)),
        "summary_row_count": int(len(summaries)),
        "node_row_count": int(len(node_rows)),
        "frontier_node_count": int(len(frontier)),
        "paths": {
            "config_rows": str(output_dir / CONFIG_ROWS_FILENAME),
            "summary_rows": str(output_dir / SUMMARY_ROWS_FILENAME_OUT),
            "node_rows": str(output_dir / NODE_ROWS_FILENAME_OUT),
            "node_frontier_rows": str(output_dir / NODE_FRONTIER_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-rows", type=Path, default=DEFAULT_TOP_ROWS)
    parser.add_argument("--attachment-dir", type=Path, default=DEFAULT_ATTACHMENT_DIR)
    parser.add_argument("--joint-bundle-dir", type=Path, default=DEFAULT_JOINT_BUNDLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument("--recovery-seed", type=int, default=0)
    parser.add_argument("--max-configs", type=int, default=6)
    parser.add_argument("--min-quality-gain", type=float, default=0.0)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    result = run_frontier(
        top_rows_path=args.top_rows,
        attachment_dir=args.attachment_dir,
        joint_bundle_dir=args.joint_bundle_dir,
        output_dir=args.output_dir,
        source_recovery_policy=args.source_recovery_policy,
        recovery_seed=args.recovery_seed,
        max_configs=args.max_configs,
        min_quality_gain=args.min_quality_gain,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
