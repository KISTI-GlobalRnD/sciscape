#!/usr/bin/env python3
"""Analyze volatile NanoClustering boundary cases against matched controls.

This script reads the membership-only volatile and stable-control boundary
packets, then produces pair-level deltas and event-threshold tables. It does
not run clustering, execute optimizer routes, promote wall/pathway claims, or
evaluate basin quality/cost.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_VOLATILE_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_volatile_boundary_cases_20260530"
DEFAULT_CONTROL_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_matched_controls_20260530"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_matched_control_delta_analysis_20260530"

VOLATILE_EVENTS_CSV = "nanoclustering_volatile_boundary_event_rows.csv"
CONTROL_EVENTS_CSV = "nanoclustering_matched_control_boundary_event_rows.csv"
MATCH_ROWS_CSV = "nanoclustering_volatile_to_stable_match_rows.csv"

PAIR_DELTA_ROWS_CSV = "nanoclustering_matched_pair_delta_rows.csv"
PAIR_DELTA_SUMMARY_CSV = "nanoclustering_matched_pair_delta_summary.csv"
THRESHOLD_TABLE_CSV = "nanoclustering_boundary_metric_threshold_table.csv"
EXCEPTION_ROWS_CSV = "nanoclustering_boundary_exception_rows.csv"
SUMMARY_JSON = "nanoclustering_matched_control_delta_summary.json"
REPORT_MD = "nanoclustering_matched_control_delta_report.md"
CONFIG_JSON = "nanoclustering_matched_control_delta_config.json"

CLAIM_BOUNDARY = (
    "Matched volatile/control endpoint-boundary delta diagnostics only; no route "
    "execution, wall/pathway promotion, basin-quality claim, cost claim, or "
    "directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_matched_control_delta_analysis"

SEVERE_PATTERNS = {"severe_split_boundary", "split_and_merge_boundary"}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    return value


def _join_sorted(values: pd.Series) -> str:
    return ";".join(str(value) for value in sorted(set(values.dropna().astype(str))))


def _join_ints(values: pd.Series) -> str:
    return ";".join(str(int(value)) for value in sorted(set(values.dropna().astype(int))))


def _pattern_count(frame: pd.DataFrame, pattern: str) -> int:
    return int(frame["boundary_pattern"].eq(pattern).sum())


def _severe_count(frame: pd.DataFrame) -> int:
    return int(frame["boundary_pattern"].isin(SEVERE_PATTERNS).sum())


def _aggregate_events(events: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (branch, ref_cluster_id), group in events.groupby(["branch", "ref_cluster_id"], sort=True):
        top = group["top_split_share_ref_weight"].astype(float)
        target = group["target_share_of_best_run_cluster_weight"].astype(float)
        split = group["split_segment_count_ge5_weight"].astype(float)
        merge = group["merge_contributor_count_ge5_weight"].astype(float)
        row = {
            "branch": str(branch),
            f"{prefix}_ref_cluster_id": int(ref_cluster_id),
            f"{prefix}_event_count": int(len(group)),
            f"{prefix}_comparison_seeds": _join_ints(group["comparison_seed"]),
            f"{prefix}_boundary_patterns": _join_sorted(group["boundary_pattern"]),
            f"{prefix}_ref_unit_count": int(group["ref_unit_count"].iloc[0]),
            f"{prefix}_ref_weight_sum": int(group["ref_weight_sum"].iloc[0]),
            f"{prefix}_top_split_share_min": float(top.min()),
            f"{prefix}_top_split_share_median": float(top.median()),
            f"{prefix}_top_split_share_mean": float(top.mean()),
            f"{prefix}_top_split_share_max": float(top.max()),
            f"{prefix}_fragmentation_index_median": float(1.0 - top.median()),
            f"{prefix}_target_run_share_min": float(target.min()),
            f"{prefix}_target_run_share_median": float(target.median()),
            f"{prefix}_target_run_share_mean": float(target.mean()),
            f"{prefix}_target_run_share_max": float(target.max()),
            f"{prefix}_absorption_index_median": float(1.0 - target.median()),
            f"{prefix}_split_segments_ge5_min": float(split.min()),
            f"{prefix}_split_segments_ge5_median": float(split.median()),
            f"{prefix}_split_segments_ge5_mean": float(split.mean()),
            f"{prefix}_split_segments_ge5_max": float(split.max()),
            f"{prefix}_merge_contributors_ge5_min": float(merge.min()),
            f"{prefix}_merge_contributors_ge5_median": float(merge.median()),
            f"{prefix}_merge_contributors_ge5_mean": float(merge.mean()),
            f"{prefix}_merge_contributors_ge5_max": float(merge.max()),
            f"{prefix}_severe_or_split_merge_event_count": _severe_count(group),
            f"{prefix}_split_and_merge_event_count": _pattern_count(group, "split_and_merge_boundary"),
            f"{prefix}_severe_split_event_count": _pattern_count(group, "severe_split_boundary"),
            f"{prefix}_merge_absorption_event_count": _pattern_count(group, "merge_absorption_boundary"),
            f"{prefix}_mild_or_label_event_count": _pattern_count(group, "mild_or_label_reassignment_boundary"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _classify_pair(row: pd.Series) -> str:
    fragmentation_delta = float(row["delta_fragmentation_index_median_volatile_minus_control"])
    absorption_delta = float(row["delta_absorption_index_median_volatile_minus_control"])
    if fragmentation_delta >= 0.40 and absorption_delta >= 0.20:
        return "volatile_fragmentation_and_absorption_gap"
    if fragmentation_delta >= 0.40:
        return "volatile_fragmentation_gap"
    if absorption_delta >= 0.20:
        return "volatile_absorption_gap"
    if (
        float(row["control_fragmentation_index_median"]) < 0.20
        and float(row["control_absorption_index_median"]) >= 0.40
    ):
        return "control_absorption_without_fragmentation"
    return "weak_or_mixed_gap"


def _pair_delta_rows(
    *,
    match_rows: pd.DataFrame,
    volatile_events: pd.DataFrame,
    control_events: pd.DataFrame,
) -> pd.DataFrame:
    volatile_agg = _aggregate_events(volatile_events, prefix="volatile")
    control_agg = _aggregate_events(control_events, prefix="control")
    rows = match_rows.merge(
        volatile_agg,
        on=["branch", "volatile_ref_cluster_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_event"),
    ).merge(
        control_agg,
        on=["branch", "control_ref_cluster_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_event"),
    )
    if rows.isna().any().any():
        missing = rows[rows.isna().any(axis=1)]
        raise ValueError(f"unmatched volatile/control aggregates: {missing[['branch', 'volatile_ref_cluster_id', 'control_ref_cluster_id']].to_dict('records')}")

    rows["delta_top_split_share_median_control_minus_volatile"] = (
        rows["control_top_split_share_median"] - rows["volatile_top_split_share_median"]
    )
    rows["delta_fragmentation_index_median_volatile_minus_control"] = (
        rows["volatile_fragmentation_index_median"] - rows["control_fragmentation_index_median"]
    )
    rows["delta_absorption_index_median_volatile_minus_control"] = (
        rows["volatile_absorption_index_median"] - rows["control_absorption_index_median"]
    )
    rows["delta_target_run_share_median_control_minus_volatile"] = (
        rows["control_target_run_share_median"] - rows["volatile_target_run_share_median"]
    )
    rows["delta_split_segments_ge5_median_volatile_minus_control"] = (
        rows["volatile_split_segments_ge5_median"] - rows["control_split_segments_ge5_median"]
    )
    rows["delta_merge_contributors_ge5_median_volatile_minus_control"] = (
        rows["volatile_merge_contributors_ge5_median"] - rows["control_merge_contributors_ge5_median"]
    )
    rows["volatile_more_fragmented"] = (
        rows["volatile_top_split_share_median"] < rows["control_top_split_share_median"]
    )
    rows["volatile_more_absorbed"] = (
        rows["volatile_target_run_share_median"] < rows["control_target_run_share_median"]
    )
    rows["volatile_more_split_segments"] = (
        rows["volatile_split_segments_ge5_median"] > rows["control_split_segments_ge5_median"]
    )
    rows["control_absorption_without_fragmentation"] = (
        rows["control_fragmentation_index_median"].lt(0.20)
        & rows["control_absorption_index_median"].ge(0.40)
    )
    rows["pair_boundary_axis"] = rows.apply(_classify_pair, axis=1)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY

    preferred = [
        "branch",
        "volatile_ref_cluster_id",
        "control_ref_cluster_id",
        "volatile_ref_weight_sum",
        "control_ref_weight_sum",
        "volatile_ref_unit_count",
        "control_ref_unit_count",
        "match_score",
        "volatile_event_count",
        "control_event_count",
        "volatile_boundary_patterns",
        "control_boundary_patterns",
        "volatile_top_split_share_median",
        "control_top_split_share_median",
        "delta_top_split_share_median_control_minus_volatile",
        "volatile_fragmentation_index_median",
        "control_fragmentation_index_median",
        "delta_fragmentation_index_median_volatile_minus_control",
        "volatile_target_run_share_median",
        "control_target_run_share_median",
        "delta_target_run_share_median_control_minus_volatile",
        "volatile_absorption_index_median",
        "control_absorption_index_median",
        "delta_absorption_index_median_volatile_minus_control",
        "volatile_split_segments_ge5_median",
        "control_split_segments_ge5_median",
        "delta_split_segments_ge5_median_volatile_minus_control",
        "volatile_merge_contributors_ge5_median",
        "control_merge_contributors_ge5_median",
        "delta_merge_contributors_ge5_median_volatile_minus_control",
        "volatile_severe_or_split_merge_event_count",
        "control_severe_or_split_merge_event_count",
        "volatile_more_fragmented",
        "volatile_more_absorbed",
        "volatile_more_split_segments",
        "control_absorption_without_fragmentation",
        "pair_boundary_axis",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(
        ["branch", "delta_fragmentation_index_median_volatile_minus_control", "volatile_ref_weight_sum"],
        ascending=[True, False, False],
    )


def _sign_test_p_two_sided(values: pd.Series) -> float | None:
    signs = values.dropna().map(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    positives = int(signs.eq(1).sum())
    negatives = int(signs.eq(-1).sum())
    n = positives + negatives
    if n == 0:
        return None
    tail = min(positives, negatives)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def _median(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].median())


def _summarize_pair_group(frame: pd.DataFrame, *, branch: str) -> dict[str, Any]:
    return {
        "branch": branch,
        "pair_count": int(len(frame)),
        "volatile_more_fragmented_pair_count": int(frame["volatile_more_fragmented"].sum()),
        "volatile_more_absorbed_pair_count": int(frame["volatile_more_absorbed"].sum()),
        "volatile_more_split_segments_pair_count": int(frame["volatile_more_split_segments"].sum()),
        "control_absorption_without_fragmentation_pair_count": int(
            frame["control_absorption_without_fragmentation"].sum()
        ),
        "volatile_severe_or_split_merge_event_count": int(
            frame["volatile_severe_or_split_merge_event_count"].sum()
        ),
        "control_severe_or_split_merge_event_count": int(
            frame["control_severe_or_split_merge_event_count"].sum()
        ),
        "volatile_top_split_share_median": _median(frame, "volatile_top_split_share_median"),
        "control_top_split_share_median": _median(frame, "control_top_split_share_median"),
        "delta_top_split_share_median_control_minus_volatile_median": _median(
            frame, "delta_top_split_share_median_control_minus_volatile"
        ),
        "delta_fragmentation_index_median_volatile_minus_control_median": _median(
            frame, "delta_fragmentation_index_median_volatile_minus_control"
        ),
        "volatile_target_run_share_median": _median(frame, "volatile_target_run_share_median"),
        "control_target_run_share_median": _median(frame, "control_target_run_share_median"),
        "delta_absorption_index_median_volatile_minus_control_median": _median(
            frame, "delta_absorption_index_median_volatile_minus_control"
        ),
        "delta_split_segments_ge5_median_volatile_minus_control_median": _median(
            frame, "delta_split_segments_ge5_median_volatile_minus_control"
        ),
        "delta_merge_contributors_ge5_median_volatile_minus_control_median": _median(
            frame, "delta_merge_contributors_ge5_median_volatile_minus_control"
        ),
        "fragmentation_delta_sign_test_p_two_sided": _sign_test_p_two_sided(
            frame["delta_fragmentation_index_median_volatile_minus_control"]
        ),
        "absorption_delta_sign_test_p_two_sided": _sign_test_p_two_sided(
            frame["delta_absorption_index_median_volatile_minus_control"]
        ),
        "split_segments_delta_sign_test_p_two_sided": _sign_test_p_two_sided(
            frame["delta_split_segments_ge5_median_volatile_minus_control"]
        ),
    }


def _pair_delta_summary(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows = [_summarize_pair_group(pair_rows, branch="all")]
    for branch, group in pair_rows.groupby("branch", sort=True):
        rows.append(_summarize_pair_group(group, branch=str(branch)))
    return pd.DataFrame(rows)


def _threshold_definitions() -> list[tuple[str, str, Any]]:
    return [
        (
            "fragmentation_top_split_lt_0p35",
            "top_split_share_ref_weight < 0.35",
            lambda frame: frame["top_split_share_ref_weight"].lt(0.35),
        ),
        (
            "fragmentation_top_split_lt_0p50",
            "top_split_share_ref_weight < 0.50",
            lambda frame: frame["top_split_share_ref_weight"].lt(0.50),
        ),
        (
            "fragmentation_top_split_lt_0p80",
            "top_split_share_ref_weight < 0.80",
            lambda frame: frame["top_split_share_ref_weight"].lt(0.80),
        ),
        (
            "high_split_segments_ge4",
            "split_segment_count_ge5_weight >= 4",
            lambda frame: frame["split_segment_count_ge5_weight"].ge(4),
        ),
        (
            "severe_or_split_merge_pattern",
            "boundary_pattern in severe/split-and-merge",
            lambda frame: frame["boundary_pattern"].isin(SEVERE_PATTERNS),
        ),
        (
            "split_and_merge_pattern",
            "boundary_pattern == split_and_merge_boundary",
            lambda frame: frame["boundary_pattern"].eq("split_and_merge_boundary"),
        ),
        (
            "target_run_share_lt_0p50",
            "target_share_of_best_run_cluster_weight < 0.50",
            lambda frame: frame["target_share_of_best_run_cluster_weight"].lt(0.50),
        ),
        (
            "target_run_share_lt_0p25",
            "target_share_of_best_run_cluster_weight < 0.25",
            lambda frame: frame["target_share_of_best_run_cluster_weight"].lt(0.25),
        ),
    ]


def _threshold_table(volatile_events: pd.DataFrame, control_events: pd.DataFrame) -> pd.DataFrame:
    volatile = volatile_events.copy()
    volatile["cohort"] = "volatile"
    control = control_events.copy()
    control["cohort"] = "stable_matched_control"
    events = pd.concat([volatile, control], ignore_index=True, sort=False)
    rows: list[dict[str, Any]] = []
    for threshold_name, expression, predicate in _threshold_definitions():
        mask = predicate(events)
        for (cohort, branch), group in events.groupby(["cohort", "branch"], sort=True):
            group_mask = mask.loc[group.index]
            rows.append(
                {
                    "threshold_name": threshold_name,
                    "threshold_expression": expression,
                    "cohort": str(cohort),
                    "branch": str(branch),
                    "event_count": int(len(group)),
                    "hit_event_count": int(group_mask.sum()),
                    "hit_rate": float(group_mask.mean()),
                    "hit_ref_cluster_count": int(group.loc[group_mask, "ref_cluster_id"].nunique()),
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "quality_cost_status": QUALITY_COST_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
        for cohort, group in events.groupby("cohort", sort=True):
            group_mask = mask.loc[group.index]
            rows.append(
                {
                    "threshold_name": threshold_name,
                    "threshold_expression": expression,
                    "cohort": str(cohort),
                    "branch": "all",
                    "event_count": int(len(group)),
                    "hit_event_count": int(group_mask.sum()),
                    "hit_rate": float(group_mask.mean()),
                    "hit_ref_cluster_count": int(group.loc[group_mask, "ref_cluster_id"].nunique()),
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "quality_cost_status": QUALITY_COST_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows).sort_values(["threshold_name", "branch", "cohort"])


def _exception_rows(volatile_events: pd.DataFrame, control_events: pd.DataFrame) -> pd.DataFrame:
    volatile = volatile_events.copy()
    volatile["cohort"] = "volatile"
    control = control_events.copy()
    control["cohort"] = "stable_matched_control"
    events = pd.concat([volatile, control], ignore_index=True, sort=False)
    exception_frames: list[pd.DataFrame] = []

    control_absorption = events[
        events["cohort"].eq("stable_matched_control")
        & events["top_split_share_ref_weight"].ge(0.80)
        & events["target_share_of_best_run_cluster_weight"].lt(0.60)
    ].copy()
    control_absorption["exception_type"] = "control_absorption_without_fragmentation"
    exception_frames.append(control_absorption)

    volatile_mild = events[
        events["cohort"].eq("volatile")
        & events["boundary_pattern"].eq("mild_or_label_reassignment_boundary")
    ].copy()
    volatile_mild["exception_type"] = "volatile_mild_or_label_reassignment"
    exception_frames.append(volatile_mild)

    volatile_intact_absorption = events[
        events["cohort"].eq("volatile")
        & events["top_split_share_ref_weight"].ge(0.80)
        & events["target_share_of_best_run_cluster_weight"].lt(0.60)
    ].copy()
    volatile_intact_absorption["exception_type"] = "volatile_absorption_without_fragmentation"
    exception_frames.append(volatile_intact_absorption)

    rows = pd.concat(exception_frames, ignore_index=True, sort=False)
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "exception_type",
                "cohort",
                "branch",
                "event_id",
                "ref_cluster_id",
                "comparison_seed",
                "boundary_pattern",
                "ref_weight_sum",
                "top_split_share_ref_weight",
                "target_share_of_best_run_cluster_weight",
                "split_segment_count_ge5_weight",
                "merge_contributor_count_ge5_weight",
                "route_execution_status",
                "wall_promotion_status",
                "quality_cost_status",
                "claim_boundary",
            ]
        )
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    columns = [
        "exception_type",
        "cohort",
        "branch",
        "event_id",
        "ref_cluster_id",
        "comparison_seed",
        "boundary_pattern",
        "ref_weight_sum",
        "top_split_share_ref_weight",
        "target_share_of_best_run_cluster_weight",
        "split_segment_count_ge5_weight",
        "merge_contributor_count_ge5_weight",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    return rows[columns].sort_values(
        ["exception_type", "cohort", "branch", "ref_weight_sum"],
        ascending=[True, True, True, False],
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in rows.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    suffix = []
    if len(frame) > max_rows:
        suffix.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    pair_rows: pd.DataFrame,
    pair_summary: pd.DataFrame,
    threshold_table: pd.DataFrame,
    exception_rows: pd.DataFrame,
) -> None:
    overall = pair_summary[pair_summary["branch"].eq("all")].iloc[0]
    severe_threshold = threshold_table[
        threshold_table["threshold_name"].eq("severe_or_split_merge_pattern")
        & threshold_table["branch"].eq("all")
    ].sort_values("cohort")
    low_top_split = threshold_table[
        threshold_table["threshold_name"].eq("fragmentation_top_split_lt_0p50")
        & threshold_table["branch"].eq("all")
    ].sort_values("cohort")
    most_fragmented = pair_rows.sort_values(
        "delta_fragmentation_index_median_volatile_minus_control",
        ascending=False,
    )
    text = [
        "# NanoClustering Matched-Control Delta Analysis",
        "",
        f"- pair_count: `{int(overall['pair_count'])}`",
        f"- volatile_more_fragmented_pair_count: `{int(overall['volatile_more_fragmented_pair_count'])}`",
        f"- volatile_more_split_segments_pair_count: `{int(overall['volatile_more_split_segments_pair_count'])}`",
        f"- control_absorption_without_fragmentation_pair_count: `{int(overall['control_absorption_without_fragmentation_pair_count'])}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair-Level Summary",
        "",
        _markdown_table(
            pair_summary,
            [
                "branch",
                "pair_count",
                "volatile_more_fragmented_pair_count",
                "volatile_more_absorbed_pair_count",
                "volatile_more_split_segments_pair_count",
                "volatile_top_split_share_median",
                "control_top_split_share_median",
                "delta_fragmentation_index_median_volatile_minus_control_median",
                "delta_absorption_index_median_volatile_minus_control_median",
                "fragmentation_delta_sign_test_p_two_sided",
            ],
            max_rows=8,
        ),
        "",
        "## Fragmentation Threshold",
        "",
        _markdown_table(
            low_top_split,
            [
                "cohort",
                "branch",
                "event_count",
                "hit_event_count",
                "hit_rate",
                "hit_ref_cluster_count",
            ],
            max_rows=8,
        ),
        "",
        "## Severe Pattern Threshold",
        "",
        _markdown_table(
            severe_threshold,
            [
                "cohort",
                "branch",
                "event_count",
                "hit_event_count",
                "hit_rate",
                "hit_ref_cluster_count",
            ],
            max_rows=8,
        ),
        "",
        "## Largest Fragmentation Gaps",
        "",
        _markdown_table(
            most_fragmented,
            [
                "branch",
                "volatile_ref_cluster_id",
                "control_ref_cluster_id",
                "volatile_ref_weight_sum",
                "control_ref_weight_sum",
                "volatile_top_split_share_median",
                "control_top_split_share_median",
                "delta_fragmentation_index_median_volatile_minus_control",
                "pair_boundary_axis",
            ],
            max_rows=12,
        ),
        "",
        "## Exceptions",
        "",
        _markdown_table(
            exception_rows,
            [
                "exception_type",
                "cohort",
                "branch",
                "event_id",
                "ref_cluster_id",
                "boundary_pattern",
                "top_split_share_ref_weight",
                "target_share_of_best_run_cluster_weight",
            ],
            max_rows=20,
        ),
        "",
        "## Read",
        "",
        "- The strongest separation is the fragmentation axis: volatile references usually have much lower top-split retention than their matched stable controls.",
        "- Absorption is not a standalone basin-boundary definition: stable controls can be mostly intact while absorbed into a larger run cluster.",
        "- The useful primitive is therefore two-axis endpoint-boundary structure, not a single quality-like scalar: fragmentation and absorption should remain separate until route traces exist.",
        "- This analysis is still endpoint cartography only; it does not establish an optimizer-native wall or traversable pathway.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def analyze(*, volatile_dir: Path, control_dir: Path, output_dir: Path) -> dict[str, Any]:
    volatile_events = _read_csv(volatile_dir / VOLATILE_EVENTS_CSV)
    control_events = _read_csv(control_dir / CONTROL_EVENTS_CSV)
    match_rows = _read_csv(control_dir / MATCH_ROWS_CSV)

    pair_rows = _pair_delta_rows(
        match_rows=match_rows,
        volatile_events=volatile_events,
        control_events=control_events,
    )
    pair_summary = _pair_delta_summary(pair_rows)
    threshold_table = _threshold_table(volatile_events, control_events)
    exception_rows = _exception_rows(volatile_events, control_events)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(pair_rows, output_dir / PAIR_DELTA_ROWS_CSV)
    _write_csv(pair_summary, output_dir / PAIR_DELTA_SUMMARY_CSV)
    _write_csv(threshold_table, output_dir / THRESHOLD_TABLE_CSV)
    _write_csv(exception_rows, output_dir / EXCEPTION_ROWS_CSV)

    summary = {
        "ok": True,
        "volatile_dir": _rel(volatile_dir),
        "control_dir": _rel(control_dir),
        "output_dir": _rel(output_dir),
        "pair_count": int(len(pair_rows)),
        "pair_summary_rows": int(len(pair_summary)),
        "threshold_rows": int(len(threshold_table)),
        "exception_rows": int(len(exception_rows)),
        "pair_boundary_axis_counts": {
            str(k): int(v) for k, v in pair_rows["pair_boundary_axis"].value_counts().to_dict().items()
        },
        "overall": _json_safe(pair_summary[pair_summary["branch"].eq("all")].iloc[0]),
        "claim_boundary": CLAIM_BOUNDARY,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "quality_cost_status": QUALITY_COST_STATUS,
        "outputs": {
            "pair_delta_rows_csv": _rel(output_dir / PAIR_DELTA_ROWS_CSV),
            "pair_delta_summary_csv": _rel(output_dir / PAIR_DELTA_SUMMARY_CSV),
            "threshold_table_csv": _rel(output_dir / THRESHOLD_TABLE_CSV),
            "exception_rows_csv": _rel(output_dir / EXCEPTION_ROWS_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "volatile_dir": str(volatile_dir),
        "control_dir": str(control_dir),
        "output_dir": str(output_dir),
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
        pair_rows=pair_rows,
        pair_summary=pair_summary,
        threshold_table=threshold_table,
        exception_rows=exception_rows,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volatile-dir", type=Path, default=DEFAULT_VOLATILE_DIR)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(
        volatile_dir=args.volatile_dir.resolve(),
        control_dir=args.control_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
