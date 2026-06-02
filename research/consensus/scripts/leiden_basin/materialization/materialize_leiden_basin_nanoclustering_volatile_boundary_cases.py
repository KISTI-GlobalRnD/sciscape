#!/usr/bin/env python3
"""Materialize volatile NanoClustering seed-boundary case packets.

This reads the external NanoClustering endpoint-landscape registry and expands
the most volatile reference clusters into split/merge case rows. It does not
run clustering, execute routes, promote wall/pathway claims, inspect
quality/cost, or change NanoClustering artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_LANDSCAPE_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_external_landscape_20260530"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_volatile_boundary_cases_20260530"

REGISTRY_CSV = "nanoclustering_external_endpoint_registry.csv"
PERSISTENCE_SUMMARY_CSV = "nanoclustering_external_reference_cluster_persistence_summary.csv"
PERSISTENCE_BY_SEED_CSV = "nanoclustering_external_reference_cluster_persistence_by_seed.csv"

SELECTED_CLUSTERS_CSV = "nanoclustering_volatile_selected_reference_clusters.csv"
EVENT_ROWS_CSV = "nanoclustering_volatile_boundary_event_rows.csv"
SPLIT_SEGMENTS_CSV = "nanoclustering_volatile_split_segments.csv"
MERGE_CONTEXT_CSV = "nanoclustering_volatile_merge_context.csv"
UNIT_SAMPLES_CSV = "nanoclustering_volatile_unit_samples.csv"
BOUNDARY_PATTERN_SUMMARY_CSV = "nanoclustering_volatile_boundary_pattern_summary.csv"
CLUSTER_REPEAT_SUMMARY_CSV = "nanoclustering_volatile_cluster_repeat_summary.csv"
SUMMARY_JSON = "nanoclustering_volatile_boundary_case_summary.json"
REPORT_MD = "nanoclustering_volatile_boundary_case_report.md"
CONFIG_JSON = "nanoclustering_volatile_boundary_case_config.json"

CLAIM_BOUNDARY = (
    "Volatile endpoint-boundary case packet only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)
QUALITY_COST_STATUS = "excluded_volatile_boundary_case_packet"
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"empty CSV: {path}") from exc


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


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


def _load_seed_membership(registry_row: pd.Series) -> pd.DataFrame:
    path = Path(str(registry_row["absolute_path"]))
    unit_col = str(registry_row["unit_col"])
    weight_col = str(registry_row["weight_col"])
    label_col = "candidate_micro_id"
    frame = pq.read_table(path, columns=[unit_col, weight_col, label_col]).to_pandas()
    frame = frame.rename(
        columns={
            unit_col: "unit_id",
            weight_col: "unit_weight",
            label_col: "cluster_id",
        }
    )
    frame["unit_id"] = frame["unit_id"].astype("int64")
    frame["unit_weight"] = frame["unit_weight"].astype("int64")
    frame["cluster_id"] = frame["cluster_id"].astype("int64")
    return frame.sort_values(["unit_id", "unit_weight"]).reset_index(drop=True)


def _select_clusters(
    persistence_summary: pd.DataFrame,
    *,
    clusters_per_branch: int,
    min_ref_weight: int,
) -> pd.DataFrame:
    eligible = persistence_summary[persistence_summary["ref_weight_sum"].ge(min_ref_weight)].copy()
    selected = []
    for branch, group in eligible.groupby("branch"):
        ranked = group.sort_values(
            [
                "best_share_ref_weight_min",
                "runs_ge80_weight",
                "ref_weight_sum",
                "ref_cluster_id",
            ],
            ascending=[True, True, False, True],
        ).head(clusters_per_branch)
        selected.append(ranked)
    if not selected:
        return pd.DataFrame()
    rows = pd.concat(selected, ignore_index=True)
    rows["selection_reason"] = (
        "lowest reference-cluster best-share across seed0-to-other-seed comparisons"
    )
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(["branch", "best_share_ref_weight_min", "ref_cluster_id"]).reset_index(
        drop=True
    )


def _select_events(
    selected_clusters: pd.DataFrame,
    persistence_by_seed: pd.DataFrame,
    *,
    seeds_per_cluster: int,
) -> pd.DataFrame:
    key_cols = ["comparability_group", "branch", "ref_cluster_id"]
    merged = persistence_by_seed.merge(
        selected_clusters[key_cols],
        on=key_cols,
        how="inner",
    )
    selected = []
    for _, group in merged.groupby(key_cols):
        selected.append(
            group.sort_values(
                ["best_share_ref_weight", "overlap_weight_sum", "comparison_seed"],
                ascending=[True, False, True],
            ).head(seeds_per_cluster)
        )
    if not selected:
        return pd.DataFrame()
    rows = pd.concat(selected, ignore_index=True)
    rows["event_id"] = rows.apply(
        lambda row: (
            f"{row['branch']}_ref{int(row['ref_cluster_id'])}_"
            f"seed{int(row['comparison_seed']):03d}"
        ),
        axis=1,
    )
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(["branch", "ref_cluster_id", "comparison_seed"]).reset_index(drop=True)


def _boundary_pattern(top_share: float, target_run_share: float) -> str:
    if top_share < 0.35 and target_run_share < 0.50:
        return "split_and_merge_boundary"
    if top_share < 0.35:
        return "severe_split_boundary"
    if target_run_share < 0.50:
        return "merge_absorption_boundary"
    if top_share < 0.60:
        return "moderate_split_boundary"
    return "mild_or_label_reassignment_boundary"


def _split_segments(
    *,
    event: pd.Series,
    aligned: pd.DataFrame,
    max_segments: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ref_cluster_id = int(event["ref_cluster_id"])
    ref = aligned[aligned["ref_cluster_id"].eq(ref_cluster_id)].copy()
    ref_unit_count = int(len(ref))
    ref_weight_sum = int(ref["unit_weight"].sum())
    segments = (
        ref.groupby("run_cluster_id", as_index=False)
        .agg(
            segment_unit_count=("unit_id", "size"),
            segment_weight_sum=("unit_weight", "sum"),
        )
        .sort_values(["segment_weight_sum", "segment_unit_count", "run_cluster_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    segments["event_id"] = str(event["event_id"])
    segments["segment_rank"] = segments.index + 1
    segments["branch"] = str(event["branch"])
    segments["comparison_seed"] = int(event["comparison_seed"])
    segments["ref_cluster_id"] = ref_cluster_id
    segments["ref_unit_count"] = ref_unit_count
    segments["ref_weight_sum"] = ref_weight_sum
    segments["segment_share_ref_units"] = segments["segment_unit_count"] / ref_unit_count
    segments["segment_share_ref_weight"] = segments["segment_weight_sum"] / ref_weight_sum
    segments["is_best_run_cluster"] = segments["run_cluster_id"].eq(int(event["best_run_cluster_id"]))
    top = segments.iloc[0]
    split_segment_count_ge5_weight = int(segments["segment_share_ref_weight"].ge(0.05).sum())
    metrics = {
        "split_segment_count": int(len(segments)),
        "split_segment_count_ge5_weight": split_segment_count_ge5_weight,
        "top_split_run_cluster_id": int(top["run_cluster_id"]),
        "top_split_weight_sum": int(top["segment_weight_sum"]),
        "top_split_share_ref_weight": float(top["segment_share_ref_weight"]),
        "top_split_share_ref_units": float(top["segment_share_ref_units"]),
    }
    segments["route_execution_status"] = ROUTE_EXECUTION_STATUS
    segments["wall_promotion_status"] = WALL_PROMOTION_STATUS
    segments["quality_cost_status"] = QUALITY_COST_STATUS
    segments["claim_boundary"] = CLAIM_BOUNDARY
    return segments.head(max_segments), metrics


def _merge_context(
    *,
    event: pd.Series,
    aligned: pd.DataFrame,
    max_contributors: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ref_cluster_id = int(event["ref_cluster_id"])
    run_cluster_id = int(event["best_run_cluster_id"])
    run = aligned[aligned["run_cluster_id"].eq(run_cluster_id)].copy()
    run_unit_count = int(len(run))
    run_weight_sum = int(run["unit_weight"].sum())
    ref_totals = aligned.groupby("ref_cluster_id", as_index=False).agg(
        contributor_ref_unit_count=("unit_id", "size"),
        contributor_ref_weight_sum=("unit_weight", "sum"),
    )
    contributors = (
        run.groupby("ref_cluster_id", as_index=False)
        .agg(
            contributor_unit_count=("unit_id", "size"),
            contributor_weight_sum=("unit_weight", "sum"),
        )
        .merge(ref_totals, on="ref_cluster_id", how="left")
        .sort_values(
            ["contributor_weight_sum", "contributor_unit_count", "ref_cluster_id"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    contributors["event_id"] = str(event["event_id"])
    contributors["contributor_rank"] = contributors.index + 1
    contributors["branch"] = str(event["branch"])
    contributors["comparison_seed"] = int(event["comparison_seed"])
    contributors["run_cluster_id"] = run_cluster_id
    contributors["run_unit_count"] = run_unit_count
    contributors["run_weight_sum"] = run_weight_sum
    contributors["target_ref_cluster_id"] = ref_cluster_id
    contributors["is_target_ref_cluster"] = contributors["ref_cluster_id"].eq(ref_cluster_id)
    contributors["contributor_share_run_units"] = (
        contributors["contributor_unit_count"] / run_unit_count
    )
    contributors["contributor_share_run_weight"] = (
        contributors["contributor_weight_sum"] / run_weight_sum
    )
    contributors["contributor_share_own_ref_weight"] = (
        contributors["contributor_weight_sum"] / contributors["contributor_ref_weight_sum"]
    )
    target = contributors[contributors["is_target_ref_cluster"]]
    target_share = float(target["contributor_share_run_weight"].iloc[0]) if not target.empty else 0.0
    metrics = {
        "merge_contributor_count": int(len(contributors)),
        "merge_contributor_count_ge5_weight": int(
            contributors["contributor_share_run_weight"].ge(0.05).sum()
        ),
        "target_share_of_best_run_cluster_weight": target_share,
        "best_run_cluster_weight_sum": run_weight_sum,
        "best_run_cluster_unit_count": run_unit_count,
    }
    contributors["route_execution_status"] = ROUTE_EXECUTION_STATUS
    contributors["wall_promotion_status"] = WALL_PROMOTION_STATUS
    contributors["quality_cost_status"] = QUALITY_COST_STATUS
    contributors["claim_boundary"] = CLAIM_BOUNDARY
    return contributors.head(max_contributors), metrics


def _unit_samples(
    *,
    event: pd.Series,
    aligned: pd.DataFrame,
    split_segments: pd.DataFrame,
    merge_context: pd.DataFrame,
    units_per_role: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_id = str(event["event_id"])
    ref_cluster_id = int(event["ref_cluster_id"])
    best_run_cluster_id = int(event["best_run_cluster_id"])
    for segment in split_segments.head(3).itertuples(index=False):
        run_cluster_id = int(segment.run_cluster_id)
        sample = aligned[
            aligned["ref_cluster_id"].eq(ref_cluster_id)
            & aligned["run_cluster_id"].eq(run_cluster_id)
        ].sort_values(["unit_weight", "unit_id"], ascending=[False, True]).head(units_per_role)
        for rank, unit in enumerate(sample.itertuples(index=False), start=1):
            rows.append(
                {
                    "event_id": event_id,
                    "sample_role": "ref_split_segment",
                    "sample_rank": rank,
                    "branch": str(event["branch"]),
                    "comparison_seed": int(event["comparison_seed"]),
                    "unit_id": int(unit.unit_id),
                    "unit_weight": int(unit.unit_weight),
                    "ref_cluster_id": int(unit.ref_cluster_id),
                    "run_cluster_id": int(unit.run_cluster_id),
                    "segment_rank": int(segment.segment_rank),
                    "is_best_run_cluster": bool(segment.is_best_run_cluster),
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "quality_cost_status": QUALITY_COST_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    for contributor in merge_context.head(3).itertuples(index=False):
        contributor_ref = int(contributor.ref_cluster_id)
        sample = aligned[
            aligned["run_cluster_id"].eq(best_run_cluster_id)
            & aligned["ref_cluster_id"].eq(contributor_ref)
        ].sort_values(["unit_weight", "unit_id"], ascending=[False, True]).head(units_per_role)
        for rank, unit in enumerate(sample.itertuples(index=False), start=1):
            rows.append(
                {
                    "event_id": event_id,
                    "sample_role": "merge_contributor",
                    "sample_rank": rank,
                    "branch": str(event["branch"]),
                    "comparison_seed": int(event["comparison_seed"]),
                    "unit_id": int(unit.unit_id),
                    "unit_weight": int(unit.unit_weight),
                    "ref_cluster_id": int(unit.ref_cluster_id),
                    "run_cluster_id": int(unit.run_cluster_id),
                    "segment_rank": "",
                    "is_best_run_cluster": True,
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "quality_cost_status": QUALITY_COST_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows)


def _materialize_cases(
    *,
    registry: pd.DataFrame,
    events: pd.DataFrame,
    max_split_segments: int,
    max_merge_contributors: int,
    units_per_role: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry_by_run = registry.set_index("run_id")
    membership_cache: dict[str, pd.DataFrame] = {}

    def load(run_id: str) -> pd.DataFrame:
        if run_id not in membership_cache:
            membership_cache[run_id] = _load_seed_membership(registry_by_run.loc[run_id])
        return membership_cache[run_id]

    event_rows: list[dict[str, Any]] = []
    split_rows: list[pd.DataFrame] = []
    merge_rows: list[pd.DataFrame] = []
    unit_rows: list[pd.DataFrame] = []

    for event in events.itertuples(index=False):
        event_series = pd.Series(event._asdict())
        ref = load(str(event.reference_run_id)).rename(columns={"cluster_id": "ref_cluster_id"})
        run = load(str(event.comparison_run_id)).rename(columns={"cluster_id": "run_cluster_id"})
        aligned = ref.merge(run, on=["unit_id", "unit_weight"], how="inner")

        splits, split_metrics = _split_segments(
            event=event_series,
            aligned=aligned,
            max_segments=max_split_segments,
        )
        merges, merge_metrics = _merge_context(
            event=event_series,
            aligned=aligned,
            max_contributors=max_merge_contributors,
        )
        samples = _unit_samples(
            event=event_series,
            aligned=aligned,
            split_segments=splits,
            merge_context=merges,
            units_per_role=units_per_role,
        )
        row = event_series.to_dict()
        row.update(split_metrics)
        row.update(merge_metrics)
        row["boundary_pattern"] = _boundary_pattern(
            float(row["top_split_share_ref_weight"]),
            float(row["target_share_of_best_run_cluster_weight"]),
        )
        row["alignment_row_count"] = int(len(aligned))
        row["route_execution_status"] = ROUTE_EXECUTION_STATUS
        row["wall_promotion_status"] = WALL_PROMOTION_STATUS
        row["quality_cost_status"] = QUALITY_COST_STATUS
        row["claim_boundary"] = CLAIM_BOUNDARY
        event_rows.append(row)
        split_rows.append(splits)
        merge_rows.append(merges)
        unit_rows.append(samples)

    return (
        pd.DataFrame(event_rows),
        pd.concat(split_rows, ignore_index=True, sort=False) if split_rows else pd.DataFrame(),
        pd.concat(merge_rows, ignore_index=True, sort=False) if merge_rows else pd.DataFrame(),
        pd.concat(unit_rows, ignore_index=True, sort=False) if unit_rows else pd.DataFrame(),
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame[columns].head(max_rows).copy().fillna("")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for record in rows.to_dict(orient="records"):
        body.append("| " + " | ".join(str(record[col]) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def _boundary_pattern_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    return (
        event_rows.groupby(["branch", "boundary_pattern"], as_index=False)
        .agg(
            event_count=("event_id", "size"),
            ref_cluster_count=("ref_cluster_id", "nunique"),
            event_ref_weight_sum=("ref_weight_sum", "sum"),
            min_top_split_share=("top_split_share_ref_weight", "min"),
            median_top_split_share=("top_split_share_ref_weight", "median"),
            max_top_split_share=("top_split_share_ref_weight", "max"),
            min_target_run_share=("target_share_of_best_run_cluster_weight", "min"),
            median_target_run_share=("target_share_of_best_run_cluster_weight", "median"),
            max_target_run_share=("target_share_of_best_run_cluster_weight", "max"),
            median_split_segments_ge5=("split_segment_count_ge5_weight", "median"),
            median_merge_contributors_ge5=("merge_contributor_count_ge5_weight", "median"),
        )
        .sort_values(["branch", "event_count"], ascending=[True, False])
    )


def _cluster_repeat_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    rows = (
        event_rows.groupby(["branch", "ref_cluster_id"], as_index=False)
        .agg(
            event_count=("event_id", "size"),
            boundary_patterns=("boundary_pattern", lambda x: ";".join(sorted(set(map(str, x))))),
            comparison_seeds=(
                "comparison_seed",
                lambda x: ",".join(f"{int(value):03d}" for value in sorted(x)),
            ),
            ref_unit_count=("ref_unit_count", "first"),
            ref_weight_sum=("ref_weight_sum", "first"),
            top_split_share_min=("top_split_share_ref_weight", "min"),
            top_split_share_max=("top_split_share_ref_weight", "max"),
            target_run_share_min=("target_share_of_best_run_cluster_weight", "min"),
            target_run_share_max=("target_share_of_best_run_cluster_weight", "max"),
            split_segments_ge5_max=("split_segment_count_ge5_weight", "max"),
            merge_contributors_ge5_max=("merge_contributor_count_ge5_weight", "max"),
        )
        .sort_values(
            ["top_split_share_min", "target_run_share_min", "ref_weight_sum"],
            ascending=[True, True, False],
        )
    )
    return rows.reset_index(drop=True)


def _write_report(
    *,
    output_dir: Path,
    selected_clusters: pd.DataFrame,
    event_rows: pd.DataFrame,
    split_segments: pd.DataFrame,
    merge_context: pd.DataFrame,
    unit_samples: pd.DataFrame,
    boundary_pattern_summary: pd.DataFrame,
    cluster_repeat_summary: pd.DataFrame,
) -> None:
    worst_events = event_rows.sort_values(
        [
            "top_split_share_ref_weight",
            "target_share_of_best_run_cluster_weight",
            "ref_weight_sum",
        ],
            ascending=[True, True, False],
    )
    high_weight_events = event_rows.sort_values(
        ["ref_weight_sum", "top_split_share_ref_weight"],
        ascending=[False, True],
    )
    text = [
        "# NanoClustering Volatile Boundary Case Packet",
        "",
        f"- selected_reference_clusters: `{len(selected_clusters)}`",
        f"- boundary_events: `{len(event_rows)}`",
        f"- split_segments: `{len(split_segments)}`",
        f"- merge_context_rows: `{len(merge_context)}`",
        f"- unit_samples: `{len(unit_samples)}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Boundary Pattern Summary",
        "",
        _markdown_table(
            boundary_pattern_summary,
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
        ),
        "",
        "## Selected Reference Clusters",
        "",
        _markdown_table(
            selected_clusters,
            [
                "branch",
                "ref_cluster_id",
                "ref_unit_count",
                "ref_weight_sum",
                "best_share_ref_weight_min",
                "best_share_ref_weight_median",
                "runs_ge80_weight",
            ],
            max_rows=24,
        ),
        "",
        "## Repeated Cluster Behavior",
        "",
        _markdown_table(
            cluster_repeat_summary,
            [
                "branch",
                "ref_cluster_id",
                "event_count",
                "boundary_patterns",
                "comparison_seeds",
                "ref_weight_sum",
                "top_split_share_min",
                "top_split_share_max",
                "target_run_share_min",
                "target_run_share_max",
                "split_segments_ge5_max",
                "merge_contributors_ge5_max",
            ],
            max_rows=24,
        ),
        "",
        "## Worst Boundary Events",
        "",
        _markdown_table(
            worst_events,
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
            max_rows=24,
        ),
        "",
        "## Largest Reference-Weight Events",
        "",
        _markdown_table(
            high_weight_events,
            [
                "event_id",
                "branch",
                "ref_cluster_id",
                "comparison_seed",
                "boundary_pattern",
                "ref_unit_count",
                "ref_weight_sum",
                "top_split_share_ref_weight",
                "target_share_of_best_run_cluster_weight",
                "split_segment_count_ge5_weight",
                "merge_contributor_count_ge5_weight",
            ],
            max_rows=16,
        ),
        "",
        "## Archetype Read",
        "",
        "- `severe_split_boundary`: the reference cluster fragments into many run clusters, while the best fragment's run cluster is mostly still from that same reference cluster.",
        "- `split_and_merge_boundary`: the reference cluster fragments, and its best fragment is absorbed into a run cluster dominated by other reference clusters.",
        "- `merge_absorption_boundary`: the reference cluster has a larger surviving fragment, but the corresponding run cluster is mostly not the target reference cluster.",
        "- `mild_or_label_reassignment_boundary`: the best fragment and target share are both high; this is the closest case to simple seed relabeling.",
        "",
        "## Read",
        "",
        "- The selected cases show endpoint boundary volatility inside clean seed ensembles.",
        "- Most selected events are split/merge or severe split cases, not simple relabeling.",
        "- These rows are evidence for basin-candidate boundary structure only; they do not show optimizer-native wall crossing.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    landscape_dir: Path,
    output_dir: Path,
    clusters_per_branch: int,
    seeds_per_cluster: int,
    min_ref_weight: int,
    max_split_segments: int,
    max_merge_contributors: int,
    units_per_role: int,
) -> dict[str, Any]:
    registry = _read_csv(landscape_dir / REGISTRY_CSV)
    persistence_summary = _read_csv(landscape_dir / PERSISTENCE_SUMMARY_CSV)
    persistence_by_seed = _read_csv(landscape_dir / PERSISTENCE_BY_SEED_CSV)

    selected_clusters = _select_clusters(
        persistence_summary,
        clusters_per_branch=clusters_per_branch,
        min_ref_weight=min_ref_weight,
    )
    events = _select_events(
        selected_clusters,
        persistence_by_seed,
        seeds_per_cluster=seeds_per_cluster,
    )
    event_rows, split_segments, merge_context, unit_samples = _materialize_cases(
        registry=registry,
        events=events,
        max_split_segments=max_split_segments,
        max_merge_contributors=max_merge_contributors,
        units_per_role=units_per_role,
    )
    boundary_pattern_summary = _boundary_pattern_summary(event_rows)
    cluster_repeat_summary = _cluster_repeat_summary(event_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(selected_clusters, output_dir / SELECTED_CLUSTERS_CSV)
    _write_csv(event_rows, output_dir / EVENT_ROWS_CSV)
    _write_csv(split_segments, output_dir / SPLIT_SEGMENTS_CSV)
    _write_csv(merge_context, output_dir / MERGE_CONTEXT_CSV)
    _write_csv(unit_samples, output_dir / UNIT_SAMPLES_CSV)
    _write_csv(boundary_pattern_summary, output_dir / BOUNDARY_PATTERN_SUMMARY_CSV)
    _write_csv(cluster_repeat_summary, output_dir / CLUSTER_REPEAT_SUMMARY_CSV)

    summary = {
        "ok": True,
        "landscape_dir": _rel(landscape_dir),
        "output_dir": _rel(output_dir),
        "selected_reference_cluster_count": int(len(selected_clusters)),
        "boundary_event_count": int(len(event_rows)),
        "split_segment_row_count": int(len(split_segments)),
        "merge_context_row_count": int(len(merge_context)),
        "unit_sample_row_count": int(len(unit_samples)),
        "boundary_pattern_summary_row_count": int(len(boundary_pattern_summary)),
        "cluster_repeat_summary_row_count": int(len(cluster_repeat_summary)),
        "branch_counts": _count(event_rows, "branch"),
        "boundary_pattern_counts": _count(event_rows, "boundary_pattern"),
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
            "boundary_pattern_summary_csv": _rel(output_dir / BOUNDARY_PATTERN_SUMMARY_CSV),
            "cluster_repeat_summary_csv": _rel(output_dir / CLUSTER_REPEAT_SUMMARY_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "landscape_dir": str(landscape_dir),
        "output_dir": str(output_dir),
        "clusters_per_branch": clusters_per_branch,
        "seeds_per_cluster": seeds_per_cluster,
        "min_ref_weight": min_ref_weight,
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
        split_segments=split_segments,
        merge_context=merge_context,
        unit_samples=unit_samples,
        boundary_pattern_summary=boundary_pattern_summary,
        cluster_repeat_summary=cluster_repeat_summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clusters-per-branch", type=int, default=12)
    parser.add_argument("--seeds-per-cluster", type=int, default=2)
    parser.add_argument("--min-ref-weight", type=int, default=3000)
    parser.add_argument("--max-split-segments", type=int, default=8)
    parser.add_argument("--max-merge-contributors", type=int, default=8)
    parser.add_argument("--units-per-role", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        landscape_dir=args.landscape_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        clusters_per_branch=args.clusters_per_branch,
        seeds_per_cluster=args.seeds_per_cluster,
        min_ref_weight=args.min_ref_weight,
        max_split_segments=args.max_split_segments,
        max_merge_contributors=args.max_merge_contributors,
        units_per_role=args.units_per_role,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
