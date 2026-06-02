#!/usr/bin/env python3
"""Materialize split/merge archetypes for a stratified fragmentation panel.

This expands a small stratified sample from the global fragmentation-boundary
inventory into event-level split/merge diagnostics. It stays inside the
NanoClustering seed-ensemble endpoint universe and does not run clustering,
execute optimizer routes, promote wall/pathway claims, or inspect basin
quality/cost.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import materialize_leiden_basin_nanoclustering_volatile_boundary_cases as boundary_cases


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_LANDSCAPE_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_external_landscape_20260530"
DEFAULT_INVENTORY_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_fragmentation_boundary_inventory_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_fragmentation_stratified_panel_20260530"
)

REGISTRY_CSV = "nanoclustering_external_endpoint_registry.csv"
PERSISTENCE_BY_SEED_CSV = "nanoclustering_external_reference_cluster_persistence_by_seed.csv"
INVENTORY_CSV = "nanoclustering_fragmentation_boundary_cluster_inventory.csv"

SELECTED_CLUSTERS_CSV = "nanoclustering_fragmentation_panel_selected_clusters.csv"
EVENT_ROWS_CSV = "nanoclustering_fragmentation_panel_boundary_event_rows.csv"
SPLIT_SEGMENTS_CSV = "nanoclustering_fragmentation_panel_split_segments.csv"
MERGE_CONTEXT_CSV = "nanoclustering_fragmentation_panel_merge_context.csv"
UNIT_SAMPLES_CSV = "nanoclustering_fragmentation_panel_unit_samples.csv"
STRATUM_PATTERN_SUMMARY_CSV = "nanoclustering_fragmentation_panel_stratum_pattern_summary.csv"
CLUSTER_REPEAT_SUMMARY_CSV = "nanoclustering_fragmentation_panel_cluster_repeat_summary.csv"
SUMMARY_JSON = "nanoclustering_fragmentation_stratified_panel_summary.json"
REPORT_MD = "nanoclustering_fragmentation_stratified_panel_report.md"
CONFIG_JSON = "nanoclustering_fragmentation_stratified_panel_config.json"

CLAIM_BOUNDARY = (
    "Stratified fragmentation endpoint-boundary panel only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_fragmentation_stratified_panel"

STRATA = [
    (
        "persistent_strong",
        "persistent_strong_fragmentation_candidate",
        6,
    ),
    (
        "recurrent_strong",
        "recurrent_strong_fragmentation_candidate",
        6,
    ),
    (
        "single_severe",
        "single_severe_fragmentation_candidate",
        4,
    ),
    (
        "single_strong",
        "single_strong_fragmentation_candidate",
        4,
    ),
    (
        "moderate",
        "moderate_fragmentation_candidate",
        4,
    ),
    (
        "stable_like",
        "matched_stable_like_reference",
        4,
    ),
]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    return boundary_cases._read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    boundary_cases._write_csv(frame, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _patch_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _rank_candidates(group: pd.DataFrame, stratum_name: str) -> pd.DataFrame:
    rows = group.copy()
    if stratum_name in {"persistent_strong", "recurrent_strong"}:
        return rows.sort_values(
            [
                "strong_fragmentation_event_count",
                "severe_fragmentation_event_count",
                "ref_weight_sum",
                "top_split_share_min",
                "ref_cluster_id",
            ],
            ascending=[False, False, False, True, True],
        )
    if stratum_name == "single_severe":
        return rows.sort_values(
            ["ref_weight_sum", "top_split_share_min", "ref_cluster_id"],
            ascending=[False, True, True],
        )
    if stratum_name == "single_strong":
        return rows.sort_values(
            ["ref_weight_sum", "top_split_share_min", "ref_cluster_id"],
            ascending=[False, True, True],
        )
    if stratum_name == "moderate":
        return rows.sort_values(
            ["ref_weight_sum", "top_split_share_min", "ref_cluster_id"],
            ascending=[False, True, True],
        )
    if stratum_name == "stable_like":
        return rows.sort_values(
            ["ref_weight_sum", "top_split_share_min", "ref_cluster_id"],
            ascending=[False, True, True],
        )
    raise ValueError(f"unknown stratum: {stratum_name}")


def _select_panel_clusters(inventory: pd.DataFrame) -> pd.DataFrame:
    selected_frames = []
    for stratum_name, rule, limit_per_branch in STRATA:
        candidates = inventory[inventory["fragmentation_boundary_rule_v0"].eq(rule)].copy()
        if candidates.empty:
            continue
        for branch, branch_group in candidates.groupby("branch", sort=True):
            ranked = _rank_candidates(branch_group, stratum_name).head(limit_per_branch).copy()
            ranked["selection_stratum"] = stratum_name
            ranked["selection_rank_in_stratum_branch"] = range(1, len(ranked) + 1)
            ranked["selection_reason"] = (
                f"top {len(ranked)} {rule} rows for branch {branch}; "
                "ranked by stratum-specific fragmentation recurrence and document weight"
            )
            selected_frames.append(ranked)
    if not selected_frames:
        return pd.DataFrame()
    rows = pd.concat(selected_frames, ignore_index=True, sort=False)
    rows = _patch_claim_columns(rows)
    return rows.sort_values(
        ["branch", "selection_stratum", "selection_rank_in_stratum_branch"]
    ).reset_index(drop=True)


def _select_panel_events(
    *,
    selected_clusters: pd.DataFrame,
    persistence_by_seed: pd.DataFrame,
    seeds_per_cluster: int,
) -> pd.DataFrame:
    key_cols = ["comparability_group", "branch", "ref_cluster_id"]
    selection_cols = key_cols + [
        "selection_stratum",
        "selection_rank_in_stratum_branch",
        "fragmentation_boundary_rule_v0",
        "top_split_share_min",
        "top_split_share_median",
        "strong_fragmentation_event_count",
        "severe_fragmentation_event_count",
        "moderate_fragmentation_event_count",
    ]
    selected = boundary_cases._select_events(
        selected_clusters[key_cols].drop_duplicates(),
        persistence_by_seed,
        seeds_per_cluster=seeds_per_cluster,
    )
    selected = selected.merge(
        selected_clusters[selection_cols],
        on=key_cols,
        how="left",
        validate="many_to_one",
    )
    if selected["selection_stratum"].isna().any():
        raise ValueError("selected event missing selection stratum")
    selected["event_id"] = selected.apply(
        lambda row: (
            f"panel_{row['selection_stratum']}_{row['branch']}_"
            f"ref{int(row['ref_cluster_id'])}_seed{int(row['comparison_seed']):03d}"
        ),
        axis=1,
    )
    selected["event_rank_for_cluster"] = selected.groupby(
        ["branch", "ref_cluster_id"], sort=False
    ).cumcount() + 1
    return _patch_claim_columns(selected)


def _attach_event_metadata(frame: pd.DataFrame, event_rows: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _patch_claim_columns(frame)
    metadata = event_rows[
        [
            "event_id",
            "selection_stratum",
            "fragmentation_boundary_rule_v0",
            "boundary_pattern",
            "top_split_share_ref_weight",
            "target_share_of_best_run_cluster_weight",
        ]
    ].drop_duplicates("event_id")
    rows = frame.merge(metadata, on="event_id", how="left", validate="many_to_one")
    return _patch_claim_columns(rows)


def _stratum_pattern_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    rows = (
        event_rows.groupby(
            ["selection_stratum", "fragmentation_boundary_rule_v0", "branch", "boundary_pattern"],
            as_index=False,
        )
        .agg(
            event_count=("event_id", "size"),
            ref_cluster_count=("ref_cluster_id", "nunique"),
            event_ref_weight_sum=("ref_weight_sum", "sum"),
            min_top_split_share=("top_split_share_ref_weight", "min"),
            median_top_split_share=("top_split_share_ref_weight", "median"),
            max_top_split_share=("top_split_share_ref_weight", "max"),
            min_target_run_share=("target_share_of_best_run_cluster_weight", "min"),
            median_target_run_share=("target_share_of_best_run_cluster_weight", "median"),
            median_split_segments_ge5=("split_segment_count_ge5_weight", "median"),
            median_merge_contributors_ge5=("merge_contributor_count_ge5_weight", "median"),
        )
        .sort_values(["selection_stratum", "branch", "event_count"], ascending=[True, True, False])
    )
    return _patch_claim_columns(rows)


def _cluster_repeat_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    rows = boundary_cases._cluster_repeat_summary(event_rows)
    if rows.empty:
        return rows
    metadata = event_rows[
        ["branch", "ref_cluster_id", "selection_stratum", "fragmentation_boundary_rule_v0"]
    ].drop_duplicates(["branch", "ref_cluster_id"])
    rows = rows.merge(metadata, on=["branch", "ref_cluster_id"], how="left", validate="one_to_one")
    preferred = [
        "selection_stratum",
        "fragmentation_boundary_rule_v0",
        "branch",
        "ref_cluster_id",
        "event_count",
        "boundary_patterns",
        "comparison_seeds",
        "ref_unit_count",
        "ref_weight_sum",
        "top_split_share_min",
        "top_split_share_max",
        "target_run_share_min",
        "target_run_share_max",
        "split_segments_ge5_max",
        "merge_contributors_ge5_max",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return _patch_claim_columns(rows[preferred + remainder])


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
    selected_clusters: pd.DataFrame,
    event_rows: pd.DataFrame,
    stratum_pattern_summary: pd.DataFrame,
    cluster_repeat_summary: pd.DataFrame,
    split_segments: pd.DataFrame,
    merge_context: pd.DataFrame,
    unit_samples: pd.DataFrame,
) -> None:
    stratum_counts = (
        selected_clusters.groupby(["selection_stratum", "branch"], as_index=False)
        .agg(
            cluster_count=("ref_cluster_id", "size"),
            ref_weight_sum=("ref_weight_sum", "sum"),
            min_top_split_share=("top_split_share_min", "min"),
            median_top_split_share=("top_split_share_median", "median"),
            median_strong_event_count=("strong_fragmentation_event_count", "median"),
        )
        .sort_values(["selection_stratum", "branch"])
    )
    event_rollup = (
        event_rows.groupby(["selection_stratum", "boundary_pattern"], as_index=False)
        .agg(
            event_count=("event_id", "size"),
            ref_cluster_count=("ref_cluster_id", "nunique"),
            median_top_split_share=("top_split_share_ref_weight", "median"),
            median_target_run_share=("target_share_of_best_run_cluster_weight", "median"),
        )
        .sort_values(["selection_stratum", "event_count"], ascending=[True, False])
    )
    worst_events = event_rows.sort_values(
        [
            "top_split_share_ref_weight",
            "target_share_of_best_run_cluster_weight",
            "ref_weight_sum",
        ],
        ascending=[True, True, False],
    )
    text = [
        "# NanoClustering Fragmentation Stratified Panel",
        "",
        f"- selected_clusters: `{len(selected_clusters)}`",
        f"- boundary_events: `{len(event_rows)}`",
        f"- split_segments: `{len(split_segments)}`",
        f"- merge_context_rows: `{len(merge_context)}`",
        f"- unit_samples: `{len(unit_samples)}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Selected Strata",
        "",
        _markdown_table(
            stratum_counts,
            [
                "selection_stratum",
                "branch",
                "cluster_count",
                "ref_weight_sum",
                "min_top_split_share",
                "median_top_split_share",
                "median_strong_event_count",
            ],
            max_rows=20,
        ),
        "",
        "## Event Pattern Rollup",
        "",
        _markdown_table(
            event_rollup,
            [
                "selection_stratum",
                "boundary_pattern",
                "event_count",
                "ref_cluster_count",
                "median_top_split_share",
                "median_target_run_share",
            ],
            max_rows=30,
        ),
        "",
        "## Stratum Pattern Summary",
        "",
        _markdown_table(
            stratum_pattern_summary,
            [
                "selection_stratum",
                "branch",
                "boundary_pattern",
                "event_count",
                "ref_cluster_count",
                "median_top_split_share",
                "median_target_run_share",
                "median_split_segments_ge5",
                "median_merge_contributors_ge5",
            ],
            max_rows=40,
        ),
        "",
        "## Repeated Cluster Behavior",
        "",
        _markdown_table(
            cluster_repeat_summary,
            [
                "selection_stratum",
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
            max_rows=40,
        ),
        "",
        "## Worst Events",
        "",
        _markdown_table(
            worst_events,
            [
                "event_id",
                "selection_stratum",
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
            max_rows=30,
        ),
        "",
        "## Read",
        "",
        "- Persistent and recurrent strong strata test whether the global fragmentation rule is supported by repeated split/merge archetypes rather than isolated bad seeds.",
        "- Single-severe and single-strong strata are intentionally included as potential edge cases; they may represent seed-specific discontinuities rather than stable basin-boundary families.",
        "- Moderate and stable-like strata are contrasts inside the same NanoClustering endpoint universe. They calibrate how quickly top-split retention turns into actual split/merge archetypes.",
        "- The panel remains endpoint-boundary cartography only; it does not establish optimizer-native wall crossing or pathway traversal.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    landscape_dir: Path,
    inventory_dir: Path,
    output_dir: Path,
    seeds_per_cluster: int,
    max_split_segments: int,
    max_merge_contributors: int,
    units_per_role: int,
) -> dict[str, Any]:
    registry = _read_csv(landscape_dir / REGISTRY_CSV)
    persistence_by_seed = _read_csv(landscape_dir / PERSISTENCE_BY_SEED_CSV)
    inventory = _read_csv(inventory_dir / INVENTORY_CSV)

    selected_clusters = _select_panel_clusters(inventory)
    events = _select_panel_events(
        selected_clusters=selected_clusters,
        persistence_by_seed=persistence_by_seed,
        seeds_per_cluster=seeds_per_cluster,
    )
    event_rows, split_segments, merge_context, unit_samples = boundary_cases._materialize_cases(
        registry=registry,
        events=events,
        max_split_segments=max_split_segments,
        max_merge_contributors=max_merge_contributors,
        units_per_role=units_per_role,
    )
    event_rows = _patch_claim_columns(event_rows)
    split_segments = _attach_event_metadata(split_segments, event_rows)
    merge_context = _attach_event_metadata(merge_context, event_rows)
    unit_samples = _attach_event_metadata(unit_samples, event_rows)
    stratum_pattern_summary = _stratum_pattern_summary(event_rows)
    cluster_repeat_summary = _cluster_repeat_summary(event_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(selected_clusters, output_dir / SELECTED_CLUSTERS_CSV)
    _write_csv(event_rows, output_dir / EVENT_ROWS_CSV)
    _write_csv(split_segments, output_dir / SPLIT_SEGMENTS_CSV)
    _write_csv(merge_context, output_dir / MERGE_CONTEXT_CSV)
    _write_csv(unit_samples, output_dir / UNIT_SAMPLES_CSV)
    _write_csv(stratum_pattern_summary, output_dir / STRATUM_PATTERN_SUMMARY_CSV)
    _write_csv(cluster_repeat_summary, output_dir / CLUSTER_REPEAT_SUMMARY_CSV)

    summary = {
        "ok": True,
        "landscape_dir": _rel(landscape_dir),
        "inventory_dir": _rel(inventory_dir),
        "output_dir": _rel(output_dir),
        "selected_cluster_count": int(len(selected_clusters)),
        "boundary_event_count": int(len(event_rows)),
        "split_segment_row_count": int(len(split_segments)),
        "merge_context_row_count": int(len(merge_context)),
        "unit_sample_row_count": int(len(unit_samples)),
        "stratum_counts": {
            str(k): int(v) for k, v in selected_clusters["selection_stratum"].value_counts().to_dict().items()
        },
        "boundary_pattern_counts": {
            str(k): int(v) for k, v in event_rows["boundary_pattern"].value_counts().to_dict().items()
        },
        "stratum_boundary_pattern_counts": {
            f"{row.selection_stratum}|{row.boundary_pattern}": int(row.event_count)
            for row in event_rows.groupby(["selection_stratum", "boundary_pattern"], as_index=False)
            .agg(event_count=("event_id", "size"))
            .itertuples(index=False)
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "quality_cost_status": QUALITY_COST_STATUS,
        "outputs": {
            "selected_clusters_csv": _rel(output_dir / SELECTED_CLUSTERS_CSV),
            "event_rows_csv": _rel(output_dir / EVENT_ROWS_CSV),
            "split_segments_csv": _rel(output_dir / SPLIT_SEGMENTS_CSV),
            "merge_context_csv": _rel(output_dir / MERGE_CONTEXT_CSV),
            "unit_samples_csv": _rel(output_dir / UNIT_SAMPLES_CSV),
            "stratum_pattern_summary_csv": _rel(output_dir / STRATUM_PATTERN_SUMMARY_CSV),
            "cluster_repeat_summary_csv": _rel(output_dir / CLUSTER_REPEAT_SUMMARY_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "landscape_dir": str(landscape_dir),
        "inventory_dir": str(inventory_dir),
        "output_dir": str(output_dir),
        "strata": [
            {
                "selection_stratum": name,
                "fragmentation_boundary_rule_v0": rule,
                "limit_per_branch": limit,
            }
            for name, rule, limit in STRATA
        ],
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
        selected_clusters=selected_clusters,
        event_rows=event_rows,
        stratum_pattern_summary=stratum_pattern_summary,
        cluster_repeat_summary=cluster_repeat_summary,
        split_segments=split_segments,
        merge_context=merge_context,
        unit_samples=unit_samples,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds-per-cluster", type=int, default=2)
    parser.add_argument("--max-split-segments", type=int, default=8)
    parser.add_argument("--max-merge-contributors", type=int, default=8)
    parser.add_argument("--units-per-role", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        landscape_dir=args.landscape_dir.resolve(),
        inventory_dir=args.inventory_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        seeds_per_cluster=args.seeds_per_cluster,
        max_split_segments=args.max_split_segments,
        max_merge_contributors=args.max_merge_contributors,
        units_per_role=args.units_per_role,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
