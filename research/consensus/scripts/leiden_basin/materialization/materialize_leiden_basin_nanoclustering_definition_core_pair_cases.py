#!/usr/bin/env python3
"""Materialize definition-core NanoClustering endpoint-pair cases.

This expands the recurrent boundary-family registry's definition-core panel
into concrete seed0-reference to comparison-seed endpoint-pair cases. It reads
memberships only. It does not run clustering, execute optimizer routes,
promote wall/pathway claims, or inspect basin quality/cost.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
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
DEFAULT_FAMILY_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_recurrent_boundary_family_registry_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_definition_core_pair_cases_20260530"
)

REGISTRY_CSV = "nanoclustering_external_endpoint_registry.csv"
FAMILY_REGISTRY_CSV = "nanoclustering_recurrent_boundary_family_registry.csv"
EVENT_SIGNATURE_ROWS_CSV = "nanoclustering_recurrent_boundary_event_signature_rows.csv"
PAIR_CONSTRUCTION_PANEL_CSV = "nanoclustering_recurrent_boundary_pair_construction_panel.csv"

SELECTED_FAMILIES_CSV = "nanoclustering_definition_core_selected_families.csv"
PAIR_EVENT_ROWS_CSV = "nanoclustering_definition_core_pair_event_rows.csv"
SPLIT_SEGMENTS_CSV = "nanoclustering_definition_core_pair_split_segments.csv"
MERGE_CONTEXT_CSV = "nanoclustering_definition_core_pair_merge_context.csv"
UNIT_SAMPLES_CSV = "nanoclustering_definition_core_pair_unit_samples.csv"
TIER_PATTERN_SUMMARY_CSV = "nanoclustering_definition_core_pair_tier_pattern_summary.csv"
FAMILY_REPEAT_SUMMARY_CSV = "nanoclustering_definition_core_pair_family_repeat_summary.csv"
SUMMARY_JSON = "nanoclustering_definition_core_pair_case_summary.json"
REPORT_MD = "nanoclustering_definition_core_pair_case_report.md"
CONFIG_JSON = "nanoclustering_definition_core_pair_case_config.json"

CLAIM_BOUNDARY = (
    "Definition-core endpoint-pair construction only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_definition_core_pair_cases"
FAMILY_SELECTION_SCOPE = "pair_panel_definition_core"
ALL_DEFINITION_CORE_SCOPE = "all_definition_core"

CORE_TIER_ORDER = {
    "repeat_severe_core": 0,
    "persistent_mixed_core": 1,
}


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


def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def _bool_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _patch_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _endpoint_pair_role(row: pd.Series) -> str:
    if bool(row["is_severe_boundary_seed"]):
        return "severe_definition_core_endpoint_pair"
    return "strong_definition_core_endpoint_pair"


def _selected_families(
    *,
    family_registry: pd.DataFrame,
    pair_panel: pd.DataFrame,
    family_selection_scope: str,
) -> pd.DataFrame:
    if family_selection_scope == FAMILY_SELECTION_SCOPE:
        selected_ids = pair_panel[pair_panel["definition_readiness"].eq("definition_core")][
            "family_id"
        ].drop_duplicates()
        rows = family_registry[family_registry["family_id"].isin(selected_ids)].copy()
        rows = rows.merge(
            pair_panel[
                [
                    "family_id",
                    "panel_rank_in_branch_tier",
                    "panel_selection_reason",
                ]
            ],
            on="family_id",
            how="left",
            validate="one_to_one",
        )
        if rows.empty:
            raise ValueError("no definition-core families selected from pair-construction panel")
        if rows["panel_rank_in_branch_tier"].isna().any():
            raise ValueError("selected definition-core family missing panel rank")
    elif family_selection_scope == ALL_DEFINITION_CORE_SCOPE:
        rows = family_registry[family_registry["definition_readiness"].eq("definition_core")].copy()
        if rows.empty:
            raise ValueError("no definition-core families found in family registry")
        rows["tier_order"] = rows["boundary_family_tier"].map(CORE_TIER_ORDER).fillna(99).astype(int)
        rows = rows.sort_values(
            [
                "branch",
                "tier_order",
                "severe_seed_count",
                "strong_seed_count",
                "ref_weight_sum",
                "top_split_share_min",
                "ref_cluster_id",
            ],
            ascending=[True, True, False, False, False, True, True],
        ).copy()
        rows["panel_rank_in_branch_tier"] = (
            rows.groupby(["branch", "boundary_family_tier"], sort=False).cumcount() + 1
        )
        rows["panel_selection_reason"] = (
            "all definition_core families from recurrent boundary registry"
        )
    else:
        raise ValueError(f"unknown family_selection_scope: {family_selection_scope}")
    rows["family_selection_scope"] = family_selection_scope
    rows["family_pair_case_status"] = "selected_for_membership_pair_case_expansion"
    rows["tier_order"] = rows["boundary_family_tier"].map(CORE_TIER_ORDER).fillna(99).astype(int)
    rows = _patch_claim_columns(rows)
    preferred = [
        "family_selection_scope",
        "family_pair_case_status",
        "panel_rank_in_branch_tier",
        "panel_selection_reason",
        "family_id",
        "branch",
        "ref_cluster_id",
        "boundary_family_tier",
        "definition_readiness",
        "expected_archetype_from_current_panel",
        "ref_unit_count",
        "ref_weight_sum",
        "strong_seed_count",
        "severe_seed_count",
        "strong_seed_list",
        "severe_seed_list",
        "top_split_share_min",
        "top_split_share_median",
        "strong_seed_target_keys",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(
        ["branch", "tier_order", "panel_rank_in_branch_tier", "ref_cluster_id"]
    )


def _select_pair_events(
    *,
    selected_families: pd.DataFrame,
    event_signatures: pd.DataFrame,
    max_events_per_family: int,
) -> pd.DataFrame:
    family_meta_cols = [
        "family_id",
        "panel_rank_in_branch_tier",
        "panel_selection_reason",
        "family_selection_scope",
        "boundary_family_tier",
        "definition_readiness",
        "expected_archetype_from_current_panel",
        "strong_seed_count",
        "severe_seed_count",
        "strong_seed_list",
        "severe_seed_list",
        "tier_order",
    ]
    rows = event_signatures[
        event_signatures["family_id"].isin(selected_families["family_id"])
        & _bool_mask(event_signatures["is_strong_boundary_seed"])
    ].copy()
    rows = rows.merge(
        selected_families[family_meta_cols],
        on="family_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_family"),
    )
    if rows.empty:
        raise ValueError("no strong endpoint-pair events selected")
    if rows["definition_readiness"].isna().any():
        raise ValueError("selected pair event missing family metadata")
    rows = rows.sort_values(
        [
            "branch",
            "tier_order",
            "panel_rank_in_branch_tier",
            "event_rank_by_fragmentation",
            "comparison_seed",
        ]
    )
    if max_events_per_family > 0:
        rows = (
            rows.groupby("family_id", group_keys=False, sort=False)
            .head(max_events_per_family)
            .reset_index(drop=True)
        )
    rows["event_id"] = rows.apply(
        lambda row: (
            f"corepair_{row['boundary_family_tier']}_{row['branch']}_"
            f"ref{int(row['ref_cluster_id'])}_seed{int(row['comparison_seed']):03d}"
        ),
        axis=1,
    )
    rows["endpoint_pair_role"] = rows.apply(_endpoint_pair_role, axis=1)
    rows["endpoint_pair_key"] = rows.apply(
        lambda row: (
            f"{row['reference_run_id']}:ref{int(row['ref_cluster_id'])}"
            f"__{row['comparison_run_id']}:run{int(row['best_run_cluster_id'])}"
        ),
        axis=1,
    )
    rows["pair_construction_status"] = "constructed_endpoint_pair_case_membership_read_only"
    rows["ref_unit_count"] = rows["ref_unit_count_family"].astype(int)
    rows["ref_weight_sum"] = rows["ref_weight_sum_family"].astype(int)
    rows["best_share_ref_weight"] = rows["top_split_share_ref_weight"].astype(float)
    rows["event_rank_for_family"] = rows.groupby("family_id", sort=False).cumcount() + 1
    rows = _patch_claim_columns(rows)
    preferred = [
        "event_id",
        "family_id",
        "family_selection_scope",
        "boundary_family_tier",
        "definition_readiness",
        "endpoint_pair_role",
        "endpoint_pair_key",
        "panel_rank_in_branch_tier",
        "event_rank_for_family",
        "event_rank_by_fragmentation",
        "event_signature_role",
        "seed_target_key",
        "comparability_group",
        "branch",
        "reference_run_id",
        "comparison_run_id",
        "reference_seed",
        "comparison_seed",
        "ref_cluster_id",
        "best_run_cluster_id",
        "ref_unit_count",
        "ref_weight_sum",
        "overlap_unit_count",
        "overlap_weight_sum",
        "best_share_ref_units",
        "best_share_ref_weight",
        "top_split_share_ref_weight",
        "fragmentation_index",
        "strong_seed_count",
        "severe_seed_count",
        "strong_seed_list",
        "severe_seed_list",
        "pair_construction_status",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].reset_index(drop=True)


def _attach_event_metadata(frame: pd.DataFrame, event_rows: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _patch_claim_columns(frame)
    metadata = event_rows[
        [
            "event_id",
            "family_id",
            "family_selection_scope",
            "boundary_family_tier",
            "definition_readiness",
            "endpoint_pair_role",
            "event_signature_role",
            "boundary_pattern",
            "top_split_share_ref_weight",
            "target_share_of_best_run_cluster_weight",
        ]
    ].drop_duplicates("event_id")
    rows = frame.merge(metadata, on="event_id", how="left", validate="many_to_one")
    return _patch_claim_columns(rows)


def _tier_pattern_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    rows = (
        event_rows.groupby(
            ["boundary_family_tier", "endpoint_pair_role", "branch", "boundary_pattern"],
            as_index=False,
        )
        .agg(
            event_count=("event_id", "size"),
            family_count=("family_id", "nunique"),
            event_ref_weight_sum=("ref_weight_sum", "sum"),
            min_top_split_share=("top_split_share_ref_weight", "min"),
            median_top_split_share=("top_split_share_ref_weight", "median"),
            max_top_split_share=("top_split_share_ref_weight", "max"),
            min_target_run_share=("target_share_of_best_run_cluster_weight", "min"),
            median_target_run_share=("target_share_of_best_run_cluster_weight", "median"),
            median_split_segments_ge5=("split_segment_count_ge5_weight", "median"),
            median_merge_contributors_ge5=("merge_contributor_count_ge5_weight", "median"),
        )
        .sort_values(
            ["boundary_family_tier", "endpoint_pair_role", "branch", "event_count"],
            ascending=[True, True, True, False],
        )
    )
    return _patch_claim_columns(rows)


def _pattern_count_text(values: pd.Series) -> str:
    counts = Counter(map(str, values))
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _family_repeat_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    rows = (
        event_rows.groupby(
            [
                "family_id",
                "boundary_family_tier",
                "definition_readiness",
                "branch",
                "ref_cluster_id",
            ],
            as_index=False,
        )
        .agg(
            event_count=("event_id", "size"),
            severe_pair_count=(
                "endpoint_pair_role",
                lambda values: int(sum(str(value).startswith("severe_") for value in values)),
            ),
            boundary_pattern_counts=("boundary_pattern", _pattern_count_text),
            comparison_seeds=(
                "comparison_seed",
                lambda values: ",".join(f"{int(value):03d}" for value in sorted(values)),
            ),
            ref_unit_count=("ref_unit_count", "first"),
            ref_weight_sum=("ref_weight_sum", "first"),
            top_split_share_min=("top_split_share_ref_weight", "min"),
            top_split_share_median=("top_split_share_ref_weight", "median"),
            target_run_share_min=("target_share_of_best_run_cluster_weight", "min"),
            target_run_share_median=("target_share_of_best_run_cluster_weight", "median"),
            split_segments_ge5_max=("split_segment_count_ge5_weight", "max"),
            merge_contributors_ge5_max=("merge_contributor_count_ge5_weight", "max"),
        )
        .sort_values(
            [
                "boundary_family_tier",
                "severe_pair_count",
                "event_count",
                "ref_weight_sum",
                "top_split_share_min",
            ],
            ascending=[True, False, False, False, True],
        )
    )
    return _patch_claim_columns(rows.reset_index(drop=True))


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
    selected_families: pd.DataFrame,
    event_rows: pd.DataFrame,
    split_segments: pd.DataFrame,
    merge_context: pd.DataFrame,
    unit_samples: pd.DataFrame,
    tier_pattern_summary: pd.DataFrame,
    family_repeat_summary: pd.DataFrame,
) -> None:
    family_counts = (
        selected_families.groupby(["boundary_family_tier", "branch"], as_index=False)
        .agg(
            family_count=("family_id", "size"),
            ref_weight_sum=("ref_weight_sum", "sum"),
            median_strong_seed_count=("strong_seed_count", "median"),
            median_severe_seed_count=("severe_seed_count", "median"),
            min_top_split_share=("top_split_share_min", "min"),
            median_top_split_share=("top_split_share_median", "median"),
        )
        .sort_values(["boundary_family_tier", "branch"])
    )
    event_rollup = (
        event_rows.groupby(["boundary_family_tier", "boundary_pattern"], as_index=False)
        .agg(
            event_count=("event_id", "size"),
            family_count=("family_id", "nunique"),
            median_top_split_share=("top_split_share_ref_weight", "median"),
            median_target_run_share=("target_share_of_best_run_cluster_weight", "median"),
        )
        .sort_values(["boundary_family_tier", "event_count"], ascending=[True, False])
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
        "# NanoClustering Definition-Core Endpoint-Pair Cases",
        "",
        f"- selected_families: `{len(selected_families)}`",
        f"- endpoint_pair_events: `{len(event_rows)}`",
        f"- split_segments: `{len(split_segments)}`",
        f"- merge_context_rows: `{len(merge_context)}`",
        f"- unit_samples: `{len(unit_samples)}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Selected Definition-Core Families",
        "",
        _markdown_table(
            family_counts,
            [
                "boundary_family_tier",
                "branch",
                "family_count",
                "ref_weight_sum",
                "median_strong_seed_count",
                "median_severe_seed_count",
                "min_top_split_share",
                "median_top_split_share",
            ],
            max_rows=12,
        ),
        "",
        "## Event Pattern Rollup",
        "",
        _markdown_table(
            event_rollup,
            [
                "boundary_family_tier",
                "boundary_pattern",
                "event_count",
                "family_count",
                "median_top_split_share",
                "median_target_run_share",
            ],
            max_rows=20,
        ),
        "",
        "## Tier Pattern Summary",
        "",
        _markdown_table(
            tier_pattern_summary,
            [
                "boundary_family_tier",
                "endpoint_pair_role",
                "branch",
                "boundary_pattern",
                "event_count",
                "family_count",
                "median_top_split_share",
                "median_target_run_share",
                "median_split_segments_ge5",
                "median_merge_contributors_ge5",
            ],
            max_rows=40,
        ),
        "",
        "## Family Repeat Summary",
        "",
        _markdown_table(
            family_repeat_summary,
            [
                "family_id",
                "boundary_family_tier",
                "branch",
                "event_count",
                "severe_pair_count",
                "boundary_pattern_counts",
                "comparison_seeds",
                "ref_weight_sum",
                "top_split_share_min",
                "target_run_share_min",
            ],
            max_rows=30,
        ),
        "",
        "## Most Fragmented Endpoint Pairs",
        "",
        _markdown_table(
            worst_events,
            [
                "event_id",
                "family_id",
                "boundary_family_tier",
                "branch",
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
        "- These rows are endpoint-pair cases for the current definition core: seed0 reference cluster versus comparison-seed best surviving run cluster.",
        "- `repeat_severe_core` tests repeatedly severe fragmentation; `persistent_mixed_core` tests persistent but less uniformly severe fragmentation.",
        "- The pair cases are the right substrate for checking whether a boundary family is internally coherent before any optimizer-native pathway or wall protocol is reopened.",
        "- This remains membership-only endpoint cartography; it does not establish route traversability, wall crossing, quality, or cost claims.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    landscape_dir: Path,
    family_dir: Path,
    output_dir: Path,
    family_selection_scope: str,
    max_events_per_family: int,
    max_split_segments: int,
    max_merge_contributors: int,
    units_per_role: int,
) -> dict[str, Any]:
    registry = _read_csv(landscape_dir / REGISTRY_CSV)
    family_registry = _read_csv(family_dir / FAMILY_REGISTRY_CSV)
    event_signatures = _read_csv(family_dir / EVENT_SIGNATURE_ROWS_CSV)
    pair_panel = _read_csv(family_dir / PAIR_CONSTRUCTION_PANEL_CSV)

    selected_families = _selected_families(
        family_registry=family_registry,
        pair_panel=pair_panel,
        family_selection_scope=family_selection_scope,
    )
    events = _select_pair_events(
        selected_families=selected_families,
        event_signatures=event_signatures,
        max_events_per_family=max_events_per_family,
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
    tier_pattern_summary = _tier_pattern_summary(event_rows)
    family_repeat_summary = _family_repeat_summary(event_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(selected_families, output_dir / SELECTED_FAMILIES_CSV)
    _write_csv(event_rows, output_dir / PAIR_EVENT_ROWS_CSV)
    _write_csv(split_segments, output_dir / SPLIT_SEGMENTS_CSV)
    _write_csv(merge_context, output_dir / MERGE_CONTEXT_CSV)
    _write_csv(unit_samples, output_dir / UNIT_SAMPLES_CSV)
    _write_csv(tier_pattern_summary, output_dir / TIER_PATTERN_SUMMARY_CSV)
    _write_csv(family_repeat_summary, output_dir / FAMILY_REPEAT_SUMMARY_CSV)

    summary = {
        "ok": True,
        "landscape_dir": _rel(landscape_dir),
        "family_dir": _rel(family_dir),
        "output_dir": _rel(output_dir),
        "family_selection_scope": family_selection_scope,
        "selected_family_count": int(len(selected_families)),
        "endpoint_pair_event_count": int(len(event_rows)),
        "split_segment_row_count": int(len(split_segments)),
        "merge_context_row_count": int(len(merge_context)),
        "unit_sample_row_count": int(len(unit_samples)),
        "tier_pattern_summary_row_count": int(len(tier_pattern_summary)),
        "family_repeat_summary_row_count": int(len(family_repeat_summary)),
        "selected_family_tier_counts": _count(selected_families, "boundary_family_tier"),
        "endpoint_pair_role_counts": _count(event_rows, "endpoint_pair_role"),
        "boundary_pattern_counts": _count(event_rows, "boundary_pattern"),
        "tier_boundary_pattern_counts": {
            f"{row.boundary_family_tier}|{row.boundary_pattern}": int(row.event_count)
            for row in event_rows.groupby(["boundary_family_tier", "boundary_pattern"], as_index=False)
            .agg(event_count=("event_id", "size"))
            .itertuples(index=False)
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "quality_cost_status": QUALITY_COST_STATUS,
        "outputs": {
            "selected_families_csv": _rel(output_dir / SELECTED_FAMILIES_CSV),
            "pair_event_rows_csv": _rel(output_dir / PAIR_EVENT_ROWS_CSV),
            "split_segments_csv": _rel(output_dir / SPLIT_SEGMENTS_CSV),
            "merge_context_csv": _rel(output_dir / MERGE_CONTEXT_CSV),
            "unit_samples_csv": _rel(output_dir / UNIT_SAMPLES_CSV),
            "tier_pattern_summary_csv": _rel(output_dir / TIER_PATTERN_SUMMARY_CSV),
            "family_repeat_summary_csv": _rel(output_dir / FAMILY_REPEAT_SUMMARY_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "landscape_dir": str(landscape_dir),
        "family_dir": str(family_dir),
        "output_dir": str(output_dir),
        "family_selection_scope": family_selection_scope,
        "max_events_per_family": max_events_per_family,
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
        selected_families=selected_families,
        event_rows=event_rows,
        split_segments=split_segments,
        merge_context=merge_context,
        unit_samples=unit_samples,
        tier_pattern_summary=tier_pattern_summary,
        family_repeat_summary=family_repeat_summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--family-dir", type=Path, default=DEFAULT_FAMILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--family-selection-scope",
        choices=[FAMILY_SELECTION_SCOPE, ALL_DEFINITION_CORE_SCOPE],
        default=FAMILY_SELECTION_SCOPE,
        help="Select the precommitted 20-family pair panel or every definition-core family.",
    )
    parser.add_argument(
        "--max-events-per-family",
        type=int,
        default=0,
        help="0 means include all strong boundary seeds for each selected family.",
    )
    parser.add_argument("--max-split-segments", type=int, default=8)
    parser.add_argument("--max-merge-contributors", type=int, default=8)
    parser.add_argument("--units-per-role", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        landscape_dir=args.landscape_dir.resolve(),
        family_dir=args.family_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        family_selection_scope=args.family_selection_scope,
        max_events_per_family=args.max_events_per_family,
        max_split_segments=args.max_split_segments,
        max_merge_contributors=args.max_merge_contributors,
        units_per_role=args.units_per_role,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
