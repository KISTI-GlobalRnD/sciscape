#!/usr/bin/env python3
"""Materialize a NanoClustering basin-vector distinction panel.

This is the second primitive basin-distinction pass. Unlike the v0 panel, it
does not reduce each event to the top endpoint handle. It records the full
significant split-segment vector and the dominant merge-host context for each
definition-core endpoint-pair event. It does not run clustering, execute
optimizer routes, promote wall/pathway claims, or inspect basin quality/cost.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_PAIR_CASE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_definition_core_pair_cases_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_basin_vector_panel_20260530"
)

PAIR_EVENT_ROWS_CSV = "nanoclustering_definition_core_pair_event_rows.csv"
SPLIT_SEGMENTS_CSV = "nanoclustering_definition_core_pair_split_segments.csv"
MERGE_CONTEXT_CSV = "nanoclustering_definition_core_pair_merge_context.csv"

EVENT_VECTOR_ROWS_CSV = "nanoclustering_basin_vector_event_rows.csv"
SEGMENT_HANDLE_ROWS_CSV = "nanoclustering_basin_vector_segment_handle_rows.csv"
HOST_CONTEXT_ROWS_CSV = "nanoclustering_basin_vector_host_context_rows.csv"
FAMILY_VECTOR_ROWS_CSV = "nanoclustering_basin_vector_family_rows.csv"
CLASS_SUMMARY_CSV = "nanoclustering_basin_vector_class_summary.csv"
SUMMARY_JSON = "nanoclustering_basin_vector_summary.json"
REPORT_MD = "nanoclustering_basin_vector_report.md"
CONFIG_JSON = "nanoclustering_basin_vector_config.json"

CLAIM_BOUNDARY = (
    "Basin-vector endpoint cartography only; no route execution, wall/pathway "
    "promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_basin_vector_panel"


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
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _count_string(values: pd.Series) -> str:
    counts = Counter(values.dropna().astype(str))
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _join_unique(values: pd.Series, *, limit: int = 30) -> str:
    seen: list[str] = []
    for value in values.dropna().astype(str):
        if value not in seen:
            seen.append(value)
    if len(seen) > limit:
        return ";".join(seen[:limit]) + f";...(+{len(seen) - limit})"
    return ";".join(seen)


def _source_handle_id(reference_run_id: str, ref_cluster_id: Any) -> str:
    return f"{reference_run_id}:ref{int(ref_cluster_id)}"


def _comparison_handle_id(comparison_run_id: str, run_cluster_id: Any) -> str:
    return f"{comparison_run_id}:run{int(run_cluster_id)}"


def _source_ref_handle_id(reference_run_id: str, ref_cluster_id: Any) -> str:
    return f"{reference_run_id}:ref{int(ref_cluster_id)}"


def _effective_count(shares: list[float]) -> float:
    denom = sum(share * share for share in shares)
    if denom <= 0:
        return 0.0
    return 1.0 / denom


def _split_vector_class(row: pd.Series) -> str:
    top1 = float(row["top1_segment_share_ref_weight"])
    top2 = float(row["top2_segment_share_ref_weight"])
    top2_sum = float(row["top2_segment_share_sum"])
    effective = float(row["effective_segment_count"])
    sig_count = int(row["significant_segment_count"])

    if top1 < 0.35 and effective >= 4.0 and sig_count >= 4:
        return "diffuse_multiway_fragmentation_vector"
    if top1 < 0.5 and top2 >= 0.35 and top2_sum >= 0.75:
        return "balanced_two_way_split_vector"
    if top1 < 0.5 and top2 >= 0.25 and sig_count >= 3:
        return "balanced_multi_handle_split_vector"
    if top1 < 0.5 and sig_count >= 3:
        return "multi_handle_fragmentation_vector"
    if top1 < 0.8 and sig_count >= 2:
        return "weak_multi_handle_boundary_vector"
    return "single_handle_or_relabeling_vector"


def _host_context_class(row: pd.Series) -> str:
    if pd.isna(row.get("dominant_host_ref_cluster_id")):
        return "host_context_missing"
    if bool(row.get("dominant_host_is_source_ref", False)):
        return "source_host_preserved"
    target_share = row.get("target_share_of_best_run_cluster_weight")
    if pd.notna(target_share) and float(target_share) < 0.5:
        return "external_host_absorption"
    return "external_host_with_high_source_share"


def _family_vector_class(group: pd.DataFrame) -> str:
    total = len(group)
    if total == 0:
        return "empty_family"
    split_counts = group["split_vector_class"].value_counts()
    host_counts = group["host_context_class"].value_counts()

    diffuse_share = split_counts.get("diffuse_multiway_fragmentation_vector", 0) / total
    balanced_two_share = split_counts.get("balanced_two_way_split_vector", 0) / total
    balanced_any_share = (
        split_counts.get("balanced_two_way_split_vector", 0)
        + split_counts.get("balanced_multi_handle_split_vector", 0)
    ) / total
    external_abs_share = host_counts.get("external_host_absorption", 0) / total
    source_host_share = host_counts.get("source_host_preserved", 0) / total

    if diffuse_share >= 0.67:
        return "diffuse_multiway_fragmentation_family"
    if balanced_two_share >= 0.67 and source_host_share >= 0.67:
        return "source_host_balanced_two_way_split_family"
    if balanced_any_share >= 0.67 and external_abs_share >= 0.67:
        return "external_host_balanced_absorption_family"
    if external_abs_share >= 0.67:
        return "external_host_absorption_family"
    if balanced_any_share >= 0.67:
        return "balanced_multi_handle_split_family"
    return "heterogeneous_basin_vector_family"


def _segment_role(row: pd.Series) -> str:
    if bool(row["is_best_run_cluster"]):
        return "top_split_segment_handle"
    if int(row["segment_rank"]) == 2:
        return "second_split_segment_handle"
    return "secondary_split_segment_handle"


def _segment_handle_rows(
    *,
    pair_events: pd.DataFrame,
    split_segments: pd.DataFrame,
    min_segment_weight: float,
    min_segment_share: float,
) -> pd.DataFrame:
    event_meta = pair_events[
        [
            "event_id",
            "family_id",
            "boundary_family_tier",
            "definition_readiness",
            "branch",
            "reference_run_id",
            "comparison_run_id",
            "reference_seed",
            "comparison_seed",
            "best_run_cluster_id",
            "boundary_pattern",
        ]
    ]
    rows = split_segments.merge(
        event_meta,
        on=[
            "event_id",
            "family_id",
            "boundary_family_tier",
            "definition_readiness",
            "branch",
            "comparison_seed",
            "boundary_pattern",
        ],
        how="left",
        validate="many_to_one",
    )
    if rows["comparison_run_id"].isna().any():
        raise ValueError("split segment row missing event metadata")
    rows = rows[
        rows["segment_weight_sum"].ge(min_segment_weight)
        & rows["segment_share_ref_weight"].ge(min_segment_share)
    ].copy()
    rows["source_basin_handle_id"] = rows.apply(
        lambda row: _source_handle_id(row["reference_run_id"], row["ref_cluster_id"]),
        axis=1,
    )
    rows["segment_basin_handle_id"] = rows.apply(
        lambda row: _comparison_handle_id(row["comparison_run_id"], row["run_cluster_id"]),
        axis=1,
    )
    rows["segment_basin_handle_role"] = rows.apply(_segment_role, axis=1)
    rows["segment_significance_status"] = "significant_split_segment_handle"
    rows["basin_identity_scope"] = "segment_endpoint_handle_local_to_run"
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY

    preferred = [
        "event_id",
        "family_id",
        "branch",
        "boundary_family_tier",
        "definition_readiness",
        "source_basin_handle_id",
        "segment_basin_handle_id",
        "segment_basin_handle_role",
        "segment_significance_status",
        "reference_run_id",
        "comparison_run_id",
        "reference_seed",
        "comparison_seed",
        "ref_cluster_id",
        "run_cluster_id",
        "segment_rank",
        "segment_unit_count",
        "segment_weight_sum",
        "segment_share_ref_units",
        "segment_share_ref_weight",
        "is_best_run_cluster",
        "boundary_pattern",
        "basin_identity_scope",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(
        ["branch", "family_id", "event_id", "segment_rank", "segment_basin_handle_id"]
    )


def _host_context_rows(
    *,
    pair_events: pd.DataFrame,
    merge_context: pd.DataFrame,
) -> pd.DataFrame:
    event_meta = pair_events[
        [
            "event_id",
            "reference_run_id",
            "comparison_run_id",
            "reference_seed",
            "comparison_seed",
            "best_run_cluster_id",
        ]
    ]
    rows = merge_context.merge(
        event_meta,
        on="event_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_event"),
    )
    if rows["comparison_run_id"].isna().any():
        raise ValueError("merge-context row missing event metadata")
    rows = rows.sort_values(["event_id", "contributor_rank"]).groupby(
        "event_id", as_index=False, sort=False
    ).head(1)
    rows = rows.rename(
        columns={
            "ref_cluster_id": "dominant_host_ref_cluster_id",
            "contributor_unit_count": "dominant_host_unit_count",
            "contributor_weight_sum": "dominant_host_weight_sum",
            "contributor_share_run_units": "dominant_host_share_run_units",
            "contributor_share_run_weight": "dominant_host_share_run_weight",
            "contributor_share_own_ref_weight": "dominant_host_share_own_ref_weight",
        }
    )
    rows["source_basin_handle_id"] = rows.apply(
        lambda row: _source_handle_id(row["reference_run_id"], row["target_ref_cluster_id"]),
        axis=1,
    )
    rows["target_run_handle_id"] = rows.apply(
        lambda row: _comparison_handle_id(row["comparison_run_id"], row["run_cluster_id"]),
        axis=1,
    )
    rows["dominant_host_handle_id"] = rows.apply(
        lambda row: _source_ref_handle_id(row["reference_run_id"], row["dominant_host_ref_cluster_id"]),
        axis=1,
    )
    rows["dominant_host_is_source_ref"] = rows["is_target_ref_cluster"].astype(bool)
    rows["host_context_class"] = rows.apply(_host_context_class, axis=1)
    rows["basin_identity_scope"] = "dominant_host_reference_handle_local_to_seed0"
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY

    preferred = [
        "event_id",
        "family_id",
        "branch",
        "boundary_family_tier",
        "definition_readiness",
        "source_basin_handle_id",
        "target_run_handle_id",
        "dominant_host_handle_id",
        "dominant_host_ref_cluster_id",
        "dominant_host_is_source_ref",
        "host_context_class",
        "dominant_host_unit_count",
        "dominant_host_weight_sum",
        "dominant_host_share_run_units",
        "dominant_host_share_run_weight",
        "dominant_host_share_own_ref_weight",
        "target_share_of_best_run_cluster_weight",
        "run_cluster_id",
        "run_unit_count",
        "run_weight_sum",
        "target_ref_cluster_id",
        "boundary_pattern",
        "basin_identity_scope",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(["branch", "family_id", "event_id"])


def _event_vector_rows(
    *,
    pair_events: pd.DataFrame,
    segment_handles: pd.DataFrame,
    host_context: pd.DataFrame,
) -> pd.DataFrame:
    segment_records: list[dict[str, Any]] = []
    for event_id, group in segment_handles.groupby("event_id", sort=False):
        ordered = group.sort_values(["segment_rank", "segment_basin_handle_id"])
        shares = ordered["segment_share_ref_weight"].astype(float).tolist()
        handles = ordered["segment_basin_handle_id"].astype(str).tolist()
        weights = ordered["segment_weight_sum"].astype(float).tolist()
        units = ordered["segment_unit_count"].astype(int).tolist()
        segment_records.append(
            {
                "event_id": event_id,
                "significant_segment_count": len(shares),
                "top1_segment_share_ref_weight": shares[0] if len(shares) >= 1 else 0.0,
                "top2_segment_share_ref_weight": shares[1] if len(shares) >= 2 else 0.0,
                "top3_segment_share_ref_weight": shares[2] if len(shares) >= 3 else 0.0,
                "top2_segment_share_sum": sum(shares[:2]),
                "top3_segment_share_sum": sum(shares[:3]),
                "significant_segment_share_sum": sum(shares),
                "effective_segment_count": _effective_count(shares),
                "top1_segment_handle_id": handles[0] if len(handles) >= 1 else "",
                "top2_segment_handle_id": handles[1] if len(handles) >= 2 else "",
                "top3_segment_handle_id": handles[2] if len(handles) >= 3 else "",
                "top_segment_handle_ids": ";".join(handles[:8]),
                "top_segment_weight_sums": ";".join(f"{value:.6g}" for value in weights[:8]),
                "top_segment_unit_counts": ";".join(str(value) for value in units[:8]),
                "top_segment_share_ref_weights": ";".join(
                    f"{value:.6g}" for value in shares[:8]
                ),
            }
        )
    segment_summary = pd.DataFrame(segment_records)
    rows = pair_events.merge(segment_summary, on="event_id", how="left", validate="one_to_one")
    if rows["significant_segment_count"].isna().any():
        raise ValueError("event missing significant segment vector")
    host_cols = [
        "event_id",
        "dominant_host_handle_id",
        "dominant_host_ref_cluster_id",
        "dominant_host_is_source_ref",
        "host_context_class",
        "dominant_host_share_run_weight",
        "dominant_host_share_own_ref_weight",
    ]
    rows = rows.merge(host_context[host_cols], on="event_id", how="left", validate="one_to_one")
    rows["source_basin_handle_id"] = rows.apply(
        lambda row: _source_handle_id(row["reference_run_id"], row["ref_cluster_id"]),
        axis=1,
    )
    rows["top1_endpoint_handle_id"] = rows.apply(
        lambda row: _comparison_handle_id(row["comparison_run_id"], row["best_run_cluster_id"]),
        axis=1,
    )
    rows["split_vector_class"] = rows.apply(_split_vector_class, axis=1)
    rows["event_vector_status"] = rows["split_vector_class"].map(
        lambda value: (
            "accepted_multi_handle_basin_vector_v1"
            if value
            not in {
                "single_handle_or_relabeling_vector",
                "weak_multi_handle_boundary_vector",
            }
            else "weak_or_single_handle_boundary_vector_v1"
        )
    )
    rows["basin_identity_scope"] = "split_segment_vector_and_dominant_host_context"
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY

    preferred = [
        "event_id",
        "family_id",
        "branch",
        "boundary_family_tier",
        "definition_readiness",
        "event_vector_status",
        "split_vector_class",
        "host_context_class",
        "source_basin_handle_id",
        "top1_endpoint_handle_id",
        "dominant_host_handle_id",
        "dominant_host_is_source_ref",
        "reference_run_id",
        "comparison_run_id",
        "reference_seed",
        "comparison_seed",
        "ref_cluster_id",
        "best_run_cluster_id",
        "boundary_pattern",
        "significant_segment_count",
        "top1_segment_share_ref_weight",
        "top2_segment_share_ref_weight",
        "top3_segment_share_ref_weight",
        "top2_segment_share_sum",
        "top3_segment_share_sum",
        "effective_segment_count",
        "target_share_of_best_run_cluster_weight",
        "dominant_host_share_run_weight",
        "dominant_host_share_own_ref_weight",
        "top_segment_handle_ids",
        "top_segment_share_ref_weights",
        "top_segment_weight_sums",
        "top_segment_unit_counts",
        "basin_identity_scope",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(["branch", "family_id", "comparison_seed"])


def _family_vector_rows(event_vectors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id, group in event_vectors.groupby("family_id", sort=False):
        first = group.iloc[0]
        total = len(group)
        split_counts = group["split_vector_class"].value_counts()
        host_counts = group["host_context_class"].value_counts()
        accepted = int(group["event_vector_status"].eq("accepted_multi_handle_basin_vector_v1").sum())
        dominant_host_counts = group["dominant_host_handle_id"].dropna().astype(str).value_counts()
        dominant_host_handle = dominant_host_counts.index[0] if len(dominant_host_counts) else ""
        dominant_host_count = int(dominant_host_counts.iloc[0]) if len(dominant_host_counts) else 0
        rows.append(
            {
                "family_id": family_id,
                "branch": first["branch"],
                "ref_cluster_id": int(first["ref_cluster_id"]),
                "boundary_family_tier": first["boundary_family_tier"],
                "definition_readiness": first["definition_readiness"],
                "event_count": total,
                "accepted_multi_handle_event_count": accepted,
                "accepted_multi_handle_event_share": accepted / total if total else 0.0,
                "split_vector_class_counts": _count_string(group["split_vector_class"]),
                "host_context_class_counts": _count_string(group["host_context_class"]),
                "family_vector_class": _family_vector_class(group),
                "comparison_seed_count": int(group["comparison_seed"].nunique()),
                "significant_segment_count_median": float(
                    group["significant_segment_count"].median()
                ),
                "significant_segment_count_max": int(group["significant_segment_count"].max()),
                "top1_segment_share_ref_weight_median": float(
                    group["top1_segment_share_ref_weight"].median()
                ),
                "top2_segment_share_ref_weight_median": float(
                    group["top2_segment_share_ref_weight"].median()
                ),
                "top2_segment_share_sum_median": float(group["top2_segment_share_sum"].median()),
                "effective_segment_count_median": float(
                    group["effective_segment_count"].median()
                ),
                "dominant_host_handle_id": dominant_host_handle,
                "dominant_host_event_count": dominant_host_count,
                "dominant_host_event_share": dominant_host_count / total if total else 0.0,
                "distinct_dominant_host_count": int(
                    group["dominant_host_handle_id"].dropna().nunique()
                ),
                "top1_endpoint_handle_count": int(group["top1_endpoint_handle_id"].nunique()),
                "top_segment_handle_signature_count": int(
                    group["top_segment_handle_ids"].nunique()
                ),
                "boundary_pattern_counts": _count_string(group["boundary_pattern"]),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["boundary_family_tier", "family_vector_class", "accepted_multi_handle_event_count"],
        ascending=[True, True, False],
    )


def _class_summary(family_vectors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (tier, family_class), group in family_vectors.groupby(
        ["boundary_family_tier", "family_vector_class"], sort=True
    ):
        rows.append(
            {
                "boundary_family_tier": tier,
                "family_vector_class": family_class,
                "family_count": int(len(group)),
                "event_count_sum": int(group["event_count"].sum()),
                "accepted_multi_handle_event_count_sum": int(
                    group["accepted_multi_handle_event_count"].sum()
                ),
                "significant_segment_count_median": float(
                    group["significant_segment_count_median"].median()
                ),
                "top1_segment_share_ref_weight_median": float(
                    group["top1_segment_share_ref_weight_median"].median()
                ),
                "top2_segment_share_ref_weight_median": float(
                    group["top2_segment_share_ref_weight_median"].median()
                ),
                "effective_segment_count_median": float(
                    group["effective_segment_count_median"].median()
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(["boundary_family_tier", "family_vector_class"])


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for _, row in rows.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
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
    event_vectors: pd.DataFrame,
    segment_handles: pd.DataFrame,
    host_context: pd.DataFrame,
    family_vectors: pd.DataFrame,
    class_summary: pd.DataFrame,
) -> None:
    source_host_balanced = int(
        family_vectors["family_vector_class"].eq(
            "source_host_balanced_two_way_split_family"
        ).sum()
    )
    diffuse = int(
        family_vectors["family_vector_class"].eq(
            "diffuse_multiway_fragmentation_family"
        ).sum()
    )
    external_abs = int(
        family_vectors["family_vector_class"].isin(
            {
                "external_host_balanced_absorption_family",
                "external_host_absorption_family",
            }
        ).sum()
    )
    text = [
        "# NanoClustering Basin Vector Panel",
        "",
        f"- event_vector_rows: `{len(event_vectors)}`",
        f"- segment_handle_rows: `{len(segment_handles)}`",
        f"- host_context_rows: `{len(host_context)}`",
        f"- family_vector_rows: `{len(family_vectors)}`",
        f"- class_summary_rows: `{len(class_summary)}`",
        f"- diffuse_multiway_fragmentation_families: `{diffuse}`",
        f"- source_host_balanced_two_way_split_families: `{source_host_balanced}`",
        f"- external_host_absorption_like_families: `{external_abs}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Class Summary",
        "",
        _markdown_table(
            class_summary,
            [
                "boundary_family_tier",
                "family_vector_class",
                "family_count",
                "event_count_sum",
                "accepted_multi_handle_event_count_sum",
                "significant_segment_count_median",
                "top1_segment_share_ref_weight_median",
                "top2_segment_share_ref_weight_median",
                "effective_segment_count_median",
            ],
            max_rows=20,
        ),
        "",
        "## Family Vectors",
        "",
        _markdown_table(
            family_vectors,
            [
                "family_id",
                "boundary_family_tier",
                "event_count",
                "family_vector_class",
                "split_vector_class_counts",
                "host_context_class_counts",
                "significant_segment_count_median",
                "top1_segment_share_ref_weight_median",
                "top2_segment_share_ref_weight_median",
                "dominant_host_event_share",
            ],
            max_rows=25,
        ),
        "",
        "## Event Vector Examples",
        "",
        _markdown_table(
            event_vectors,
            [
                "event_id",
                "family_id",
                "boundary_family_tier",
                "split_vector_class",
                "host_context_class",
                "significant_segment_count",
                "top1_segment_share_ref_weight",
                "top2_segment_share_ref_weight",
                "effective_segment_count",
            ],
            max_rows=25,
        ),
        "",
        "## Read",
        "",
        "- The v1 object is a split-segment vector plus dominant host context, not a single top endpoint handle.",
        "- Most repeat-severe families resolve into diffuse multiway fragmentation families.",
        "- Several persistent-mixed families are better described as external-host absorption or source-host balanced split families.",
        "- This remains endpoint cartography and does not define final attraction basins or wall/pathway claims.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    pair_case_dir: Path,
    output_dir: Path,
    min_segment_weight: float,
    min_segment_share: float,
) -> dict[str, Any]:
    pair_events = _read_csv(pair_case_dir / PAIR_EVENT_ROWS_CSV)
    split_segments = _read_csv(pair_case_dir / SPLIT_SEGMENTS_CSV)
    merge_context = _read_csv(pair_case_dir / MERGE_CONTEXT_CSV)

    segment_handles = _segment_handle_rows(
        pair_events=pair_events,
        split_segments=split_segments,
        min_segment_weight=min_segment_weight,
        min_segment_share=min_segment_share,
    )
    host_context = _host_context_rows(pair_events=pair_events, merge_context=merge_context)
    event_vectors = _event_vector_rows(
        pair_events=pair_events,
        segment_handles=segment_handles,
        host_context=host_context,
    )
    family_vectors = _family_vector_rows(event_vectors)
    class_summary = _class_summary(family_vectors)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(event_vectors, output_dir / EVENT_VECTOR_ROWS_CSV)
    _write_csv(segment_handles, output_dir / SEGMENT_HANDLE_ROWS_CSV)
    _write_csv(host_context, output_dir / HOST_CONTEXT_ROWS_CSV)
    _write_csv(family_vectors, output_dir / FAMILY_VECTOR_ROWS_CSV)
    _write_csv(class_summary, output_dir / CLASS_SUMMARY_CSV)
    _write_report(
        output_dir=output_dir,
        event_vectors=event_vectors,
        segment_handles=segment_handles,
        host_context=host_context,
        family_vectors=family_vectors,
        class_summary=class_summary,
    )

    summary = {
        "ok": True,
        "pair_case_dir": _rel(pair_case_dir),
        "output_dir": _rel(output_dir),
        "event_vector_row_count": int(len(event_vectors)),
        "segment_handle_row_count": int(len(segment_handles)),
        "host_context_row_count": int(len(host_context)),
        "family_vector_row_count": int(len(family_vectors)),
        "class_summary_row_count": int(len(class_summary)),
        "accepted_multi_handle_event_count": int(
            event_vectors["event_vector_status"].eq(
                "accepted_multi_handle_basin_vector_v1"
            ).sum()
        ),
        "family_vector_class_counts": {
            str(key): int(value)
            for key, value in family_vectors["family_vector_class"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "split_vector_class_counts": {
            str(key): int(value)
            for key, value in event_vectors["split_vector_class"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "host_context_class_counts": {
            str(key): int(value)
            for key, value in event_vectors["host_context_class"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "min_segment_weight": min_segment_weight,
        "min_segment_share": min_segment_share,
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "event_vector_rows_csv": _rel(output_dir / EVENT_VECTOR_ROWS_CSV),
            "segment_handle_rows_csv": _rel(output_dir / SEGMENT_HANDLE_ROWS_CSV),
            "host_context_rows_csv": _rel(output_dir / HOST_CONTEXT_ROWS_CSV),
            "family_vector_rows_csv": _rel(output_dir / FAMILY_VECTOR_ROWS_CSV),
            "class_summary_csv": _rel(output_dir / CLASS_SUMMARY_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "pair_case_dir": _rel(pair_case_dir),
        "output_dir": _rel(output_dir),
        "min_segment_weight": min_segment_weight,
        "min_segment_share": min_segment_share,
        "split_vector_rules": {
            "diffuse_multiway_fragmentation_vector": "top1 < 0.35 and effective_segment_count >= 4.0 and significant_segment_count >= 4",
            "balanced_two_way_split_vector": "top1 < 0.5 and top2 >= 0.35 and top2_sum >= 0.75",
            "balanced_multi_handle_split_vector": "top1 < 0.5 and top2 >= 0.25 and significant_segment_count >= 3",
            "multi_handle_fragmentation_vector": "top1 < 0.5 and significant_segment_count >= 3",
        },
        "host_context_rules": {
            "source_host_preserved": "dominant merge contributor is the source reference cluster",
            "external_host_absorption": "dominant merge contributor is not source and source share in target run cluster < 0.5",
            "external_host_with_high_source_share": "dominant merge contributor is not source and source share in target run cluster >= 0.5",
        },
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
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-case-dir", type=Path, default=DEFAULT_PAIR_CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-segment-weight", type=float, default=5.0)
    parser.add_argument("--min-segment-share", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        pair_case_dir=args.pair_case_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        min_segment_weight=args.min_segment_weight,
        min_segment_share=args.min_segment_share,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
