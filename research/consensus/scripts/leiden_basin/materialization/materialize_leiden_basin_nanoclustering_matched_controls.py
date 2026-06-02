#!/usr/bin/env python3
"""Materialize stable matched controls for NanoClustering boundary cases.

This script matches each volatile seed0 reference cluster to a stable reference
cluster with similar branch, document weight, and unit count, then computes the
same split/merge boundary packet. It does not run clustering, execute routes,
promote wall/pathway claims, inspect basin quality/cost, or change
NanoClustering artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import materialize_leiden_basin_nanoclustering_volatile_boundary_cases as volatile_cases


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_LANDSCAPE_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_external_landscape_20260530"
DEFAULT_VOLATILE_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_volatile_boundary_cases_20260530"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_matched_controls_20260530"

SELECTED_CONTROLS_CSV = "nanoclustering_matched_control_selected_reference_clusters.csv"
MATCH_ROWS_CSV = "nanoclustering_volatile_to_stable_match_rows.csv"
CONTROL_EVENT_ROWS_CSV = "nanoclustering_matched_control_boundary_event_rows.csv"
CONTROL_SPLIT_SEGMENTS_CSV = "nanoclustering_matched_control_split_segments.csv"
CONTROL_MERGE_CONTEXT_CSV = "nanoclustering_matched_control_merge_context.csv"
CONTROL_UNIT_SAMPLES_CSV = "nanoclustering_matched_control_unit_samples.csv"
CONTROL_PATTERN_SUMMARY_CSV = "nanoclustering_matched_control_boundary_pattern_summary.csv"
CONTROL_CLUSTER_REPEAT_SUMMARY_CSV = "nanoclustering_matched_control_cluster_repeat_summary.csv"
VOLATILE_CONTROL_EVENT_SUMMARY_CSV = "nanoclustering_volatile_vs_matched_control_event_summary.csv"
SUMMARY_JSON = "nanoclustering_matched_control_summary.json"
REPORT_MD = "nanoclustering_matched_control_report.md"
CONFIG_JSON = "nanoclustering_matched_control_config.json"

CLAIM_BOUNDARY = (
    "Matched stable-control boundary diagnostics only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)
QUALITY_COST_STATUS = "excluded_matched_control_boundary_packet"
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    return volatile_cases._read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    volatile_cases._write_csv(frame, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def _patch_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    for col, value in (
        ("route_execution_status", ROUTE_EXECUTION_STATUS),
        ("wall_promotion_status", WALL_PROMOTION_STATUS),
        ("quality_cost_status", QUALITY_COST_STATUS),
        ("claim_boundary", CLAIM_BOUNDARY),
    ):
        if col in rows:
            rows[col] = value
    return rows


def _match_score(volatile_row: pd.Series, candidate_row: pd.Series) -> float:
    v_weight = max(float(volatile_row["ref_weight_sum"]), 1.0)
    c_weight = max(float(candidate_row["ref_weight_sum"]), 1.0)
    v_units = max(float(volatile_row["ref_unit_count"]), 1.0)
    c_units = max(float(candidate_row["ref_unit_count"]), 1.0)
    weight_term = abs(math.log(c_weight / v_weight))
    unit_term = abs(math.log(c_units / v_units))
    stability_term = max(0.0, 1.0 - float(candidate_row["best_share_ref_weight_min"])) * 0.10
    return weight_term + 0.35 * unit_term + stability_term


def _select_matched_controls(
    *,
    volatile_selected: pd.DataFrame,
    persistence_summary: pd.DataFrame,
    stable_min_share: float,
    stable_runs_ge80: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    volatile_keys = set(zip(volatile_selected["branch"], volatile_selected["ref_cluster_id"]))
    stable_pool = persistence_summary[
        persistence_summary["best_share_ref_weight_min"].ge(stable_min_share)
        & persistence_summary["runs_ge80_weight"].ge(stable_runs_ge80)
    ].copy()
    selected_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()

    for volatile_row in volatile_selected.sort_values(
        ["branch", "best_share_ref_weight_min", "ref_weight_sum"],
        ascending=[True, True, False],
    ).itertuples(index=False):
        v = pd.Series(volatile_row._asdict())
        candidates = stable_pool[stable_pool["branch"].eq(v["branch"])].copy()
        candidates = candidates[
            ~candidates.apply(
                lambda row: (str(row["branch"]), int(row["ref_cluster_id"])) in volatile_keys
                or (str(row["branch"]), int(row["ref_cluster_id"])) in used,
                axis=1,
            )
        ]
        if candidates.empty:
            raise ValueError(f"no stable control candidate for {v['branch']} ref {v['ref_cluster_id']}")
        candidates["match_score"] = candidates.apply(lambda row: _match_score(v, row), axis=1)
        control = candidates.sort_values(
            ["match_score", "ref_weight_sum", "ref_cluster_id"],
            ascending=[True, False, True],
        ).iloc[0]
        key = (str(control["branch"]), int(control["ref_cluster_id"]))
        used.add(key)
        selected = control.to_dict()
        selected.update(
            {
                "selection_role": "stable_matched_control",
                "matched_volatile_ref_cluster_id": int(v["ref_cluster_id"]),
                "matched_volatile_ref_weight_sum": int(v["ref_weight_sum"]),
                "matched_volatile_ref_unit_count": int(v["ref_unit_count"]),
                "match_score": float(control["match_score"]),
                "selection_reason": (
                    "stable all-seed reference cluster matched by branch, "
                    "log document weight, and log unit count"
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        selected_rows.append(selected)
        match_rows.append(
            {
                "branch": str(v["branch"]),
                "volatile_ref_cluster_id": int(v["ref_cluster_id"]),
                "control_ref_cluster_id": int(control["ref_cluster_id"]),
                "volatile_ref_unit_count": int(v["ref_unit_count"]),
                "control_ref_unit_count": int(control["ref_unit_count"]),
                "volatile_ref_weight_sum": int(v["ref_weight_sum"]),
                "control_ref_weight_sum": int(control["ref_weight_sum"]),
                "volatile_min_best_share_ref_weight": float(v["best_share_ref_weight_min"]),
                "control_min_best_share_ref_weight": float(control["best_share_ref_weight_min"]),
                "volatile_runs_ge80_weight": int(v["runs_ge80_weight"]),
                "control_runs_ge80_weight": int(control["runs_ge80_weight"]),
                "match_score": float(control["match_score"]),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    selected_controls = pd.DataFrame(selected_rows).sort_values(["branch", "matched_volatile_ref_cluster_id"])
    match_rows_frame = pd.DataFrame(match_rows).sort_values(["branch", "volatile_ref_cluster_id"])
    return selected_controls.reset_index(drop=True), match_rows_frame.reset_index(drop=True)


def _select_control_events(
    *,
    selected_controls: pd.DataFrame,
    persistence_by_seed: pd.DataFrame,
    seeds_per_cluster: int,
) -> pd.DataFrame:
    event_input = selected_controls.copy()
    event_input["selection_reason"] = event_input.get("selection_reason", "")
    events = volatile_cases._select_events(
        event_input,
        persistence_by_seed,
        seeds_per_cluster=seeds_per_cluster,
    )
    events["event_id"] = events.apply(
        lambda row: (
            f"control_{row['branch']}_ref{int(row['ref_cluster_id'])}_"
            f"seed{int(row['comparison_seed']):03d}"
        ),
        axis=1,
    )
    return _patch_claim_columns(events)


def _materialize_control_cases(
    *,
    registry: pd.DataFrame,
    events: pd.DataFrame,
    max_split_segments: int,
    max_merge_contributors: int,
    units_per_role: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows, split_segments, merge_context, unit_samples = volatile_cases._materialize_cases(
        registry=registry,
        events=events,
        max_split_segments=max_split_segments,
        max_merge_contributors=max_merge_contributors,
        units_per_role=units_per_role,
    )
    return (
        _patch_claim_columns(event_rows),
        _patch_claim_columns(split_segments),
        _patch_claim_columns(merge_context),
        _patch_claim_columns(unit_samples),
    )


def _event_summary(volatile_events: pd.DataFrame, control_events: pd.DataFrame) -> pd.DataFrame:
    v = volatile_events.copy()
    c = control_events.copy()
    v["cohort"] = "volatile"
    c["cohort"] = "stable_matched_control"
    rows = pd.concat([v, c], ignore_index=True, sort=False)
    return (
        rows.groupby(["cohort", "branch", "boundary_pattern"], as_index=False)
        .agg(
            event_count=("event_id", "size"),
            ref_cluster_count=("ref_cluster_id", "nunique"),
            ref_weight_sum_total=("ref_weight_sum", "sum"),
            top_split_share_min=("top_split_share_ref_weight", "min"),
            top_split_share_median=("top_split_share_ref_weight", "median"),
            top_split_share_mean=("top_split_share_ref_weight", "mean"),
            target_run_share_min=("target_share_of_best_run_cluster_weight", "min"),
            target_run_share_median=("target_share_of_best_run_cluster_weight", "median"),
            split_segments_ge5_median=("split_segment_count_ge5_weight", "median"),
            merge_contributors_ge5_median=("merge_contributor_count_ge5_weight", "median"),
        )
        .sort_values(["cohort", "branch", "event_count"], ascending=[True, True, False])
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    return volatile_cases._markdown_table(frame, columns, max_rows=max_rows)


def _write_report(
    *,
    output_dir: Path,
    selected_controls: pd.DataFrame,
    match_rows: pd.DataFrame,
    control_events: pd.DataFrame,
    control_pattern_summary: pd.DataFrame,
    control_cluster_repeat_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> None:
    worst_controls = control_events.sort_values(
        ["top_split_share_ref_weight", "target_share_of_best_run_cluster_weight", "ref_weight_sum"],
        ascending=[True, True, False],
    )
    text = [
        "# NanoClustering Matched Stable Controls",
        "",
        f"- selected_controls: `{len(selected_controls)}`",
        f"- control_boundary_events: `{len(control_events)}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Volatile-To-Control Matches",
        "",
        _markdown_table(
            match_rows,
            [
                "branch",
                "volatile_ref_cluster_id",
                "control_ref_cluster_id",
                "volatile_ref_weight_sum",
                "control_ref_weight_sum",
                "volatile_min_best_share_ref_weight",
                "control_min_best_share_ref_weight",
                "match_score",
            ],
            max_rows=24,
        ),
        "",
        "## Control Boundary Pattern Summary",
        "",
        _markdown_table(
            control_pattern_summary,
            [
                "branch",
                "boundary_pattern",
                "event_count",
                "ref_cluster_count",
                "event_ref_weight_sum",
                "min_top_split_share",
                "median_top_split_share",
                "min_target_run_share",
                "median_target_run_share",
            ],
            max_rows=16,
        ),
        "",
        "## Volatile Vs Stable-Control Event Summary",
        "",
        _markdown_table(
            event_summary,
            [
                "cohort",
                "branch",
                "boundary_pattern",
                "event_count",
                "ref_cluster_count",
                "top_split_share_min",
                "top_split_share_median",
                "target_run_share_median",
                "split_segments_ge5_median",
            ],
            max_rows=24,
        ),
        "",
        "## Worst Control Events",
        "",
        _markdown_table(
            worst_controls,
            [
                "event_id",
                "branch",
                "ref_cluster_id",
                "comparison_seed",
                "boundary_pattern",
                "ref_weight_sum",
                "top_split_share_ref_weight",
                "target_share_of_best_run_cluster_weight",
                "split_segment_count_ge5_weight",
                "merge_contributor_count_ge5_weight",
            ],
            max_rows=16,
        ),
        "",
        "## Repeated Control Behavior",
        "",
        _markdown_table(
            control_cluster_repeat_summary,
            [
                "branch",
                "ref_cluster_id",
                "event_count",
                "boundary_patterns",
                "comparison_seeds",
                "ref_weight_sum",
                "top_split_share_min",
                "target_run_share_min",
                "split_segments_ge5_max",
                "merge_contributors_ge5_max",
            ],
            max_rows=24,
        ),
        "",
        "## Read",
        "",
        "- Stable matched controls preserve the same endpoint and branch universe but select reference clusters that remain high-overlap across all seeds.",
        "- If controls collapse mostly to mild or moderate patterns, the volatile boundary packet is not just a size/selection artifact.",
        "- These controls still provide endpoint-boundary diagnostics only; no optimizer-native route or wall claim is introduced.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    landscape_dir: Path,
    volatile_dir: Path,
    output_dir: Path,
    stable_min_share: float,
    stable_runs_ge80: int,
    seeds_per_cluster: int,
    max_split_segments: int,
    max_merge_contributors: int,
    units_per_role: int,
) -> dict[str, Any]:
    registry = _read_csv(landscape_dir / volatile_cases.REGISTRY_CSV)
    persistence_summary = _read_csv(landscape_dir / volatile_cases.PERSISTENCE_SUMMARY_CSV)
    persistence_by_seed = _read_csv(landscape_dir / volatile_cases.PERSISTENCE_BY_SEED_CSV)
    volatile_selected = _read_csv(volatile_dir / "nanoclustering_volatile_selected_reference_clusters.csv")
    volatile_events = _read_csv(volatile_dir / "nanoclustering_volatile_boundary_event_rows.csv")

    selected_controls, match_rows = _select_matched_controls(
        volatile_selected=volatile_selected,
        persistence_summary=persistence_summary,
        stable_min_share=stable_min_share,
        stable_runs_ge80=stable_runs_ge80,
    )
    control_events_seed_rows = _select_control_events(
        selected_controls=selected_controls,
        persistence_by_seed=persistence_by_seed,
        seeds_per_cluster=seeds_per_cluster,
    )
    control_events, split_segments, merge_context, unit_samples = _materialize_control_cases(
        registry=registry,
        events=control_events_seed_rows,
        max_split_segments=max_split_segments,
        max_merge_contributors=max_merge_contributors,
        units_per_role=units_per_role,
    )
    control_pattern_summary = _patch_claim_columns(
        volatile_cases._boundary_pattern_summary(control_events)
    )
    control_cluster_repeat_summary = _patch_claim_columns(
        volatile_cases._cluster_repeat_summary(control_events)
    )
    event_summary = _event_summary(volatile_events, control_events)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(selected_controls, output_dir / SELECTED_CONTROLS_CSV)
    _write_csv(match_rows, output_dir / MATCH_ROWS_CSV)
    _write_csv(control_events, output_dir / CONTROL_EVENT_ROWS_CSV)
    _write_csv(split_segments, output_dir / CONTROL_SPLIT_SEGMENTS_CSV)
    _write_csv(merge_context, output_dir / CONTROL_MERGE_CONTEXT_CSV)
    _write_csv(unit_samples, output_dir / CONTROL_UNIT_SAMPLES_CSV)
    _write_csv(control_pattern_summary, output_dir / CONTROL_PATTERN_SUMMARY_CSV)
    _write_csv(control_cluster_repeat_summary, output_dir / CONTROL_CLUSTER_REPEAT_SUMMARY_CSV)
    _write_csv(event_summary, output_dir / VOLATILE_CONTROL_EVENT_SUMMARY_CSV)

    summary = {
        "ok": True,
        "landscape_dir": _rel(landscape_dir),
        "volatile_dir": _rel(volatile_dir),
        "output_dir": _rel(output_dir),
        "selected_control_count": int(len(selected_controls)),
        "match_row_count": int(len(match_rows)),
        "control_boundary_event_count": int(len(control_events)),
        "control_split_segment_row_count": int(len(split_segments)),
        "control_merge_context_row_count": int(len(merge_context)),
        "control_unit_sample_row_count": int(len(unit_samples)),
        "control_boundary_pattern_counts": _count(control_events, "boundary_pattern"),
        "event_summary_rows": int(len(event_summary)),
        "claim_boundary": CLAIM_BOUNDARY,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "quality_cost_status": QUALITY_COST_STATUS,
        "outputs": {
            "selected_controls_csv": _rel(output_dir / SELECTED_CONTROLS_CSV),
            "match_rows_csv": _rel(output_dir / MATCH_ROWS_CSV),
            "control_event_rows_csv": _rel(output_dir / CONTROL_EVENT_ROWS_CSV),
            "control_split_segments_csv": _rel(output_dir / CONTROL_SPLIT_SEGMENTS_CSV),
            "control_merge_context_csv": _rel(output_dir / CONTROL_MERGE_CONTEXT_CSV),
            "control_unit_samples_csv": _rel(output_dir / CONTROL_UNIT_SAMPLES_CSV),
            "control_pattern_summary_csv": _rel(output_dir / CONTROL_PATTERN_SUMMARY_CSV),
            "control_cluster_repeat_summary_csv": _rel(output_dir / CONTROL_CLUSTER_REPEAT_SUMMARY_CSV),
            "volatile_control_event_summary_csv": _rel(output_dir / VOLATILE_CONTROL_EVENT_SUMMARY_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "landscape_dir": str(landscape_dir),
        "volatile_dir": str(volatile_dir),
        "output_dir": str(output_dir),
        "stable_min_share": stable_min_share,
        "stable_runs_ge80": stable_runs_ge80,
        "seeds_per_cluster": seeds_per_cluster,
        "max_split_segments": max_split_segments,
        "max_merge_contributors": max_merge_contributors,
        "units_per_role": units_per_role,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        selected_controls=selected_controls,
        match_rows=match_rows,
        control_events=control_events,
        control_pattern_summary=control_pattern_summary,
        control_cluster_repeat_summary=control_cluster_repeat_summary,
        event_summary=event_summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--volatile-dir", type=Path, default=DEFAULT_VOLATILE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stable-min-share", type=float, default=0.8)
    parser.add_argument("--stable-runs-ge80", type=int, default=9)
    parser.add_argument("--seeds-per-cluster", type=int, default=2)
    parser.add_argument("--max-split-segments", type=int, default=8)
    parser.add_argument("--max-merge-contributors", type=int, default=8)
    parser.add_argument("--units-per-role", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        landscape_dir=args.landscape_dir.resolve(),
        volatile_dir=args.volatile_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        stable_min_share=args.stable_min_share,
        stable_runs_ge80=args.stable_runs_ge80,
        seeds_per_cluster=args.seeds_per_cluster,
        max_split_segments=args.max_split_segments,
        max_merge_contributors=args.max_merge_contributors,
        units_per_role=args.units_per_role,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
