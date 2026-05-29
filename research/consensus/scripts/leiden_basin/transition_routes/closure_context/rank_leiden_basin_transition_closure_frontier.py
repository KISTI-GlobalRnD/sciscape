#!/usr/bin/env python3
"""Rank closure labels before any basin-transition mutation.

This is a diagnostic-only frontier builder. It joins closure-context label rows
with boundary node roles, then selects small auditable closure labels for a
future closure-split shrink pilot.
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
)
from analyze_leiden_basin_transition_closure_context import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_CLOSURE_CONTEXT_DIR,
    LABEL_ROWS_FILENAME,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "basin_transition_closure_frontier_field34_cc"
)
FRONTIER_ROWS_FILENAME = "closure_label_frontier_rows.csv"
SUMMARY_FILENAME = "closure_label_frontier_summary.json"
REPORT_FILENAME = "closure_label_frontier_report.md"

PAIR_COLUMNS = [
    "case",
    "field",
    "method",
    "candidate_index",
    "vanilla_seed",
    "vanilla_randomness",
    "vanilla_requested_n_iterations",
]
ROLE_NAMES = ("collateral_like", "ambiguous", "bridge_like")
MODE_LABEL_COLUMNS = {
    "baseline_label": "baseline_label",
    "candidate_label": "candidate_label",
    "vanilla_label_source": "vanilla_label",
}

def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())

def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or denominator <= 0.0:
        return math.nan
    return float(numerator) / float(denominator)

def direct_role_features(
    *,
    label_rows: pd.DataFrame,
    node_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate boundary role features for direct nodes in each closure label."""
    if label_rows.empty:
        return label_rows.copy()
    direct = node_rows[node_rows["support_class"].eq("vanilla_extra")].copy()
    feature_frames: list[pd.DataFrame] = []
    for mode, label_column in MODE_LABEL_COLUMNS.items():
        mode_nodes = direct.copy()
        mode_nodes["closure_mode"] = mode
        mode_nodes["closure_label"] = pd.to_numeric(
            mode_nodes[label_column],
            errors="coerce",
        ).astype("Int64")
        group_cols = [*PAIR_COLUMNS, "closure_mode", "closure_label"]
        role_counts = (
            mode_nodes.groupby([*group_cols, "boundary_role"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for role in ROLE_NAMES:
            if role not in role_counts.columns:
                role_counts[role] = 0
        role_counts = role_counts.rename(
            columns={role: f"direct_{role}_nodes" for role in ROLE_NAMES}
        )
        score_cols = [
            "node_weight",
            "incident_weight_total",
            "bridge_score",
            "collateral_score",
            "necessity_score",
            "core_pull",
            "vanilla_extra_pull",
            "baseline_pull",
            "candidate_pull",
            "vanilla_pull",
            "boundary_role_margin",
        ]
        aggregations: dict[str, tuple[str, str]] = {
            "direct_node_weight_sum": ("node_weight", "sum"),
            "direct_incident_weight_sum": ("incident_weight_total", "sum"),
        }
        aggregations.update(
            {
                f"direct_{column}_mean": (column, "mean")
                for column in score_cols
                if column not in {"node_weight", "incident_weight_total"}
            }
        )
        scores = (
            mode_nodes.groupby(group_cols, dropna=False)
            .agg(**aggregations)
            .reset_index()
        )
        feature_frames.append(role_counts.merge(scores, on=group_cols, how="outer"))

    features = pd.concat(feature_frames, ignore_index=True)
    merged = label_rows.copy()
    merged["closure_label"] = pd.to_numeric(
        merged["closure_label"],
        errors="coerce",
    ).astype("Int64")
    merged = merged.merge(
        features,
        on=[*PAIR_COLUMNS, "closure_mode", "closure_label"],
        how="left",
    )
    for role in ROLE_NAMES:
        column = f"direct_{role}_nodes"
        if column not in merged.columns:
            merged[column] = 0
        merged[column] = merged[column].fillna(0).astype(int)
    merged["direct_role_covered_nodes"] = merged[
        [f"direct_{role}_nodes" for role in ROLE_NAMES]
    ].sum(axis=1)
    merged["direct_collateralish_nodes"] = (
        merged["direct_collateral_like_nodes"] + merged["direct_ambiguous_nodes"]
    )
    merged["direct_collateralish_fraction"] = [
        _safe_ratio(collateralish, direct_count)
        for collateralish, direct_count in zip(
            merged["direct_collateralish_nodes"],
            merged["direct_node_count"],
            strict=False,
        )
    ]
    merged["direct_bridge_fraction"] = [
        _safe_ratio(bridge, direct_count)
        for bridge, direct_count in zip(
            merged["direct_bridge_like_nodes"],
            merged["direct_node_count"],
            strict=False,
        )
    ]
    return merged

def _frontier_reason(row: pd.Series, config: dict[str, Any]) -> str:
    if row["closure_mode"] not in set(config["closure_modes"]):
        return "reject_mode"
    if int(row["direct_node_count"]) < int(config["min_direct_nodes"]):
        return "reject_small_direct"
    if int(row["direct_node_count"]) > int(config["max_direct_nodes"]):
        return "reject_large_direct"
    if int(row["closure_node_count"]) > int(config["max_closure_nodes"]):
        return "reject_large_closure"
    if int(row["closure_context_extra_count"]) < int(config["min_context_extra"]):
        return "reject_low_context"
    if float(row["direct_bridge_fraction"]) > float(config["max_bridge_fraction"]):
        return "reject_bridge_heavy"
    if float(row["direct_collateralish_fraction"]) < float(
        config["min_collateralish_fraction"]
    ):
        return "reject_low_collateralish"
    return "eligible"

def score_frontier_rows(
    rows: pd.DataFrame,
    *,
    closure_modes: tuple[str, ...] = ("candidate_label",),
    min_direct_nodes: int = 1,
    max_direct_nodes: int = 32,
    max_closure_nodes: int = 300,
    min_context_extra: int = 20,
    max_bridge_fraction: float = 0.25,
    min_collateralish_fraction: float = 0.5,
    top_labels_per_pair: int = 10,
) -> pd.DataFrame:
    """Score and mark closure labels for the next shrink-frontier pilot."""
    if rows.empty:
        return rows.copy()
    out = rows.copy()
    config = {
        "closure_modes": tuple(closure_modes),
        "min_direct_nodes": int(min_direct_nodes),
        "max_direct_nodes": int(max_direct_nodes),
        "max_closure_nodes": int(max_closure_nodes),
        "min_context_extra": int(min_context_extra),
        "max_bridge_fraction": float(max_bridge_fraction),
        "min_collateralish_fraction": float(min_collateralish_fraction),
    }
    for column in [
        "closure_context_ratio",
        "direct_collateralish_fraction",
        "direct_bridge_fraction",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["frontier_reason"] = [
        _frontier_reason(row, config) for _, row in out.iterrows()
    ]
    out["frontier_eligible"] = out["frontier_reason"].eq("eligible")
    out["frontier_score"] = (
        0.35 * np.log1p(out["closure_context_extra_count"].astype(float))
        + 0.25 * np.log1p(out["closure_outside_support_count"].astype(float))
        + 0.20 * np.log1p(out["closure_context_ratio"].astype(float))
        + 0.15 * out["direct_collateralish_fraction"].astype(float)
        - 0.35 * out["direct_bridge_fraction"].astype(float)
        - 0.05 * np.log1p(out["direct_node_count"].astype(float))
    )
    sort_cols = [
        *PAIR_COLUMNS,
        "closure_mode",
        "frontier_score",
        "closure_context_extra_count",
        "direct_node_count",
    ]
    out = out.sort_values(
        sort_cols,
        ascending=[True, True, True, True, True, True, True, True, False, False, False],
    ).reset_index(drop=True)
    eligible = out[out["frontier_eligible"]].copy()
    if eligible.empty:
        out["frontier_rank_in_pair"] = math.nan
        out["frontier_selected"] = False
        return out
    eligible["frontier_rank_in_pair"] = (
        eligible.groupby([*PAIR_COLUMNS, "closure_mode"])["frontier_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    out = out.merge(
        eligible[[*PAIR_COLUMNS, "closure_mode", "closure_label", "frontier_rank_in_pair"]],
        on=[*PAIR_COLUMNS, "closure_mode", "closure_label"],
        how="left",
    )
    out["frontier_selected"] = (
        out["frontier_eligible"]
        & out["frontier_rank_in_pair"].le(int(top_labels_per_pair)).fillna(False)
    )
    out.loc[
        out["frontier_eligible"] & ~out["frontier_selected"],
        "frontier_reason",
    ] = "eligible_outside_pair_topk"
    return out.sort_values(
        [
            "frontier_selected",
            *PAIR_COLUMNS,
            "closure_mode",
            "frontier_rank_in_pair",
            "frontier_score",
        ],
        ascending=[False, *([True] * (len(PAIR_COLUMNS) + 2)), False],
    ).reset_index(drop=True)

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

def write_report(path: Path, rows: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [
        "# Closure Label Frontier",
        "",
        "This diagnostic ranks closure labels before any membership mutation.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "frontier_rows",
        "selected_rows",
        "eligible_rows",
        "selected_direct_nodes_sum",
        "selected_closure_nodes_sum",
        "selected_context_extra_sum",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Selected Labels", ""])
    selected = rows[rows["frontier_selected"]].copy()
    display_cols = [
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "closure_mode",
        "closure_label",
        "frontier_rank_in_pair",
        "frontier_score",
        "direct_node_count",
        "closure_node_count",
        "closure_context_extra_count",
        "closure_context_ratio",
        "direct_collateral_like_nodes",
        "direct_ambiguous_nodes",
        "direct_bridge_like_nodes",
        "direct_collateralish_fraction",
        "direct_bridge_fraction",
    ]
    lines.extend(_markdown_table(selected[[c for c in display_cols if c in selected.columns]], max_rows=40))
    lines.extend(["", "## Rejection Summary", ""])
    if not rows.empty:
        rejects = (
            rows.groupby(["closure_mode", "frontier_reason"], as_index=False)
            .agg(labels=("closure_label", "size"))
            .sort_values(["closure_mode", "frontier_reason"])
        )
        lines.extend(_markdown_table(rejects, max_rows=40))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- `frontier_selected` means suitable for a first dry-run pilot, not accepted as an algorithmic improvement.",
            "- High closure ratio is treated as a split/merge warning, not as automatic rejection.",
            "- The next mutation pilot must still compare against vanilla, candidate, baseline, and seed controls.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_analysis(
    *,
    closure_context_dir: Path,
    boundary_dir: Path,
    output_dir: Path,
    closure_modes: tuple[str, ...],
    min_direct_nodes: int,
    max_direct_nodes: int,
    max_closure_nodes: int,
    min_context_extra: int,
    max_bridge_fraction: float,
    min_collateralish_fraction: float,
    top_labels_per_pair: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(closure_context_dir / LABEL_ROWS_FILENAME)
    nodes = pd.read_csv(boundary_dir / NODE_ROWS_FILENAME)
    rows = direct_role_features(label_rows=labels, node_rows=nodes)
    rows = score_frontier_rows(
        rows,
        closure_modes=closure_modes,
        min_direct_nodes=min_direct_nodes,
        max_direct_nodes=max_direct_nodes,
        max_closure_nodes=max_closure_nodes,
        min_context_extra=min_context_extra,
        max_bridge_fraction=max_bridge_fraction,
        min_collateralish_fraction=min_collateralish_fraction,
        top_labels_per_pair=top_labels_per_pair,
    )
    rows.to_csv(output_dir / FRONTIER_ROWS_FILENAME, index=False)
    selected = rows[rows["frontier_selected"]]
    summary = {
        "schema": "leiden_basin_transition_closure_frontier.v1",
        "closure_context_dir": str(closure_context_dir),
        "boundary_dir": str(boundary_dir),
        "output_dir": str(output_dir),
        "closure_modes": list(closure_modes),
        "frontier_rows": int(len(rows)),
        "eligible_rows": int(rows["frontier_eligible"].sum()) if not rows.empty else 0,
        "selected_rows": int(len(selected)),
        "top_labels_per_pair": int(top_labels_per_pair),
        "min_direct_nodes": int(min_direct_nodes),
        "max_direct_nodes": int(max_direct_nodes),
        "max_closure_nodes": int(max_closure_nodes),
        "min_context_extra": int(min_context_extra),
        "max_bridge_fraction": float(max_bridge_fraction),
        "min_collateralish_fraction": float(min_collateralish_fraction),
        "selected_direct_nodes_sum": (
            int(selected["direct_node_count"].sum()) if not selected.empty else 0
        ),
        "selected_closure_nodes_sum": (
            int(selected["closure_node_count"].sum()) if not selected.empty else 0
        ),
        "selected_context_extra_sum": (
            int(selected["closure_context_extra_count"].sum())
            if not selected.empty
            else 0
        ),
    }
    if not selected.empty:
        summary["selected_median_context_ratio"] = float(
            selected["closure_context_ratio"].median()
        )
        summary["selected_max_context_ratio"] = float(
            selected["closure_context_ratio"].max()
        )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / REPORT_FILENAME, rows, summary)
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure-context-dir",
        type=Path,
        default=DEFAULT_CLOSURE_CONTEXT_DIR,
    )
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--closure-modes", default="candidate_label")
    parser.add_argument("--min-direct-nodes", type=int, default=1)
    parser.add_argument("--max-direct-nodes", type=int, default=32)
    parser.add_argument("--max-closure-nodes", type=int, default=300)
    parser.add_argument("--min-context-extra", type=int, default=20)
    parser.add_argument("--max-bridge-fraction", type=float, default=0.25)
    parser.add_argument("--min-collateralish-fraction", type=float, default=0.5)
    parser.add_argument("--top-labels-per-pair", type=int, default=10)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        closure_context_dir=args.closure_context_dir,
        boundary_dir=args.boundary_dir,
        output_dir=args.output_dir,
        closure_modes=_parse_csv_tuple(args.closure_modes),
        min_direct_nodes=args.min_direct_nodes,
        max_direct_nodes=args.max_direct_nodes,
        max_closure_nodes=args.max_closure_nodes,
        min_context_extra=args.min_context_extra,
        max_bridge_fraction=args.max_bridge_fraction,
        min_collateralish_fraction=args.min_collateralish_fraction,
        top_labels_per_pair=args.top_labels_per_pair,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
