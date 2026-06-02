#!/usr/bin/env python3
"""Decompose NanoClustering definition-core v1 refinement queues.

This reads the full definition-core v1 registry and event-level coherence rows,
then tests whether non-coherent families become coherent after support-local
subfamily partitioning. It does not run clustering, execute optimizer routes,
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


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_REGISTRY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v1_family_registry_20260530"
)
DEFAULT_COHERENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_full_basin_vector_coherence_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v1_refinement_queue_decomposition_20260530"
)

V1_FAMILY_REGISTRY_CSV = "nanoclustering_definition_core_v1_family_registry.csv"
COHERENCE_EVENT_ROWS_CSV = "nanoclustering_basin_vector_coherence_event_rows.csv"

SUBFAMILY_EVENT_ROWS_CSV = "nanoclustering_definition_core_v1_refinement_subfamily_event_rows.csv"
SUBFAMILY_ROWS_CSV = "nanoclustering_definition_core_v1_refinement_subfamily_rows.csv"
FAMILY_DECOMPOSITION_ROWS_CSV = (
    "nanoclustering_definition_core_v1_refinement_family_decomposition_rows.csv"
)
AXIS_COMPARISON_ROWS_CSV = (
    "nanoclustering_definition_core_v1_refinement_axis_comparison_rows.csv"
)
CLASS_SUMMARY_CSV = "nanoclustering_definition_core_v1_refinement_class_summary.csv"
RECOVERED_SHORTLIST_CSV = (
    "nanoclustering_definition_core_v1_refinement_recovered_subfamily_shortlist.csv"
)
SUMMARY_JSON = "nanoclustering_definition_core_v1_refinement_queue_summary.json"
REPORT_MD = "nanoclustering_definition_core_v1_refinement_queue_report.md"
CONFIG_JSON = "nanoclustering_definition_core_v1_refinement_queue_config.json"

ACCEPTED_STATUS = "definition_core_v1_coherent"
CLAIM_BOUNDARY = (
    "Definition-core v1 refinement decomposition only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_definition_core_v1_refinement_queue"

AXIS_COLUMNS = {
    "split_vector_class": ["split_vector_class"],
    "host_context_class": ["host_context_class"],
    "shape_core_signature": ["shape_core_signature"],
    "host_signature": ["host_signature"],
    "boundary_pattern": ["boundary_pattern"],
    "shape_core_and_host_context": ["shape_core_signature", "host_context_class"],
    "shape_core_and_host_signature": ["shape_core_signature", "host_signature"],
    "shape_core_and_boundary_pattern": ["shape_core_signature", "boundary_pattern"],
}

STATUS_TO_PRIMARY_AXIS = {
    "definition_core_v1_numeric_stress": "split_vector_class",
    "split_coherent_host_variable_subfamily": "host_context_class",
    "host_coherent_split_mixed_subfamily": "split_vector_class",
    "heterogeneous_rule_edge_review": "split_vector_class",
}


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


def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts(dropna=False).sort_index().to_dict().items()
    }


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


def _dominant(values: pd.Series) -> tuple[str, int, float]:
    counts = Counter(values.dropna().astype(str))
    if not counts:
        return "", 0, 0.0
    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    total = len(values.dropna())
    return label, int(count), count / total if total else 0.0


def _iqr(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.quantile(0.75) - values.quantile(0.25))


def _coherence_status(row: dict[str, Any]) -> str:
    split_purity = float(row["dominant_split_vector_class_share"])
    host_purity = float(row["dominant_host_context_class_share"])
    shape_core_share = float(row["dominant_shape_core_signature_share"])
    host_share = float(row["dominant_host_handle_share"])
    top1_iqr = float(row["top1_segment_share_iqr"])
    top2_iqr = float(row["top2_segment_share_iqr"])

    split_coherent = split_purity >= 0.75 and shape_core_share >= 0.5 and top1_iqr <= 0.12
    host_coherent = host_purity >= 0.75 and host_share >= 0.5
    numeric_stable = top1_iqr <= 0.08 and top2_iqr <= 0.10

    if split_coherent and host_coherent and numeric_stable:
        return "coherent_vector_and_host_subfamily"
    if split_coherent and host_coherent:
        return "coherent_class_with_numeric_variation_subfamily"
    if split_coherent and not host_coherent:
        return "split_coherent_host_variable_subfamily"
    if host_coherent and not split_coherent:
        return "host_coherent_split_mixed_subfamily"
    return "heterogeneous_or_rule_edge_subfamily"


def _subfamily_vector_class(group: pd.DataFrame) -> str:
    total = len(group)
    if total == 0:
        return "empty_subfamily"
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
        return "diffuse_multiway_fragmentation_subfamily"
    if balanced_two_share >= 0.67 and source_host_share >= 0.67:
        return "source_host_balanced_two_way_split_subfamily"
    if balanced_any_share >= 0.67 and external_abs_share >= 0.67:
        return "external_host_balanced_absorption_subfamily"
    if external_abs_share >= 0.67:
        return "external_host_absorption_subfamily"
    if balanced_any_share >= 0.67:
        return "balanced_multi_handle_split_subfamily"
    return "heterogeneous_basin_vector_subfamily"


def _axis_key(row: pd.Series, axis: str) -> str:
    return "||".join(str(row[column]) for column in AXIS_COLUMNS[axis])


def _axis_key_label(axis: str) -> str:
    return "+".join(AXIS_COLUMNS[axis])


def _subfamily_row(
    *,
    group: pd.DataFrame,
    registry_row: pd.Series,
    decomposition_axis: str,
    decomposition_key: str,
    subfamily_rank: int,
    min_subfamily_events: int,
) -> dict[str, Any]:
    first = group.iloc[0]
    split_label, split_count, split_share = _dominant(group["split_vector_class"])
    host_label, host_count, host_share = _dominant(group["host_context_class"])
    shape_label, shape_count, shape_share = _dominant(group["shape_signature"])
    shape_core_label, shape_core_count, shape_core_share = _dominant(
        group["shape_core_signature"]
    )
    host_handle, host_handle_count, host_handle_share = _dominant(
        group["dominant_host_handle_id"]
    )
    top1 = group["top1_segment_share_ref_weight"].astype(float)
    top2 = group["top2_segment_share_ref_weight"].astype(float)
    effective = group["effective_segment_count"].astype(float)
    segment_count = group["significant_segment_count"].astype(float)
    row: dict[str, Any] = {
        "source_family_id": str(registry_row["family_id"]),
        "subfamily_id": f"{registry_row['family_id']}__{decomposition_axis}__sub{subfamily_rank:02d}",
        "decomposition_axis": decomposition_axis,
        "decomposition_axis_columns": _axis_key_label(decomposition_axis),
        "decomposition_key": decomposition_key,
        "branch": first["branch"],
        "boundary_family_tier": first["boundary_family_tier"],
        "source_definition_core_v1_status": registry_row["definition_core_v1_status"],
        "source_family_vector_class": registry_row["family_vector_class"],
        "subfamily_vector_class": _subfamily_vector_class(group),
        "event_count": int(len(group)),
        "event_count_share_of_source_family": int(len(group)) / float(registry_row["event_count"]),
        "dominant_split_vector_class": split_label,
        "dominant_split_vector_class_count": split_count,
        "dominant_split_vector_class_share": split_share,
        "split_vector_class_counts": _count_string(group["split_vector_class"]),
        "dominant_host_context_class": host_label,
        "dominant_host_context_class_count": host_count,
        "dominant_host_context_class_share": host_share,
        "host_context_class_counts": _count_string(group["host_context_class"]),
        "dominant_shape_signature": shape_label,
        "dominant_shape_signature_count": shape_count,
        "dominant_shape_signature_share": shape_share,
        "shape_signature_count": int(group["shape_signature"].nunique()),
        "dominant_shape_core_signature": shape_core_label,
        "dominant_shape_core_signature_count": shape_core_count,
        "dominant_shape_core_signature_share": shape_core_share,
        "shape_core_signature_count": int(group["shape_core_signature"].nunique()),
        "dominant_host_handle_id": host_handle,
        "dominant_host_handle_count": host_handle_count,
        "dominant_host_handle_share": host_handle_share,
        "distinct_dominant_host_handle_count": int(group["dominant_host_handle_id"].nunique()),
        "top1_segment_share_median": float(top1.median()),
        "top1_segment_share_iqr": _iqr(top1),
        "top2_segment_share_median": float(top2.median()),
        "top2_segment_share_iqr": _iqr(top2),
        "top2_segment_share_sum_median": float(
            group["top2_segment_share_sum"].astype(float).median()
        ),
        "effective_segment_count_median": float(effective.median()),
        "effective_segment_count_iqr": _iqr(effective),
        "significant_segment_count_median": float(segment_count.median()),
        "significant_segment_count_iqr": _iqr(segment_count),
        "boundary_pattern_counts": _count_string(group["boundary_pattern"]),
        "comparison_seeds": _join_unique(group["comparison_seed"]),
        "source_ref_weight_sum": int(registry_row["ref_weight_sum"]),
        "source_event_count": int(registry_row["event_count"]),
        "source_next_definition_action": registry_row["next_definition_action"],
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "quality_cost_status": QUALITY_COST_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    row["subfamily_coherence_status"] = _coherence_status(row)
    if int(row["event_count"]) < min_subfamily_events:
        row["definition_refinement_result"] = "diagnostic_tiny_subfamily_not_promoted"
    elif row["subfamily_coherence_status"] == "coherent_vector_and_host_subfamily":
        row["definition_refinement_result"] = "recovered_coherent_endpoint_vector_subfamily"
    elif (
        row["subfamily_coherence_status"]
        == "coherent_class_with_numeric_variation_subfamily"
    ):
        row["definition_refinement_result"] = "recovered_numeric_stress_subfamily"
    else:
        row["definition_refinement_result"] = "still_requires_definition_refinement"
    return row


def _subfamilies_for_axis(
    *,
    registry_row: pd.Series,
    family_events: pd.DataFrame,
    axis: str,
    min_subfamily_events: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keyed = family_events.copy()
    keyed["decomposition_key"] = keyed.apply(lambda row: _axis_key(row, axis), axis=1)
    size_order = (
        keyed.groupby("decomposition_key", as_index=False)
        .agg(event_count=("event_id", "size"))
        .sort_values(["event_count", "decomposition_key"], ascending=[False, True])
    )
    for rank, key in enumerate(size_order["decomposition_key"], start=1):
        group = keyed[keyed["decomposition_key"].eq(key)]
        rows.append(
            _subfamily_row(
                group=group,
                registry_row=registry_row,
                decomposition_axis=axis,
                decomposition_key=key,
                subfamily_rank=rank,
                min_subfamily_events=min_subfamily_events,
            )
        )
    return pd.DataFrame(rows)


def _axis_comparison_row(
    *,
    registry_row: pd.Series,
    family_events: pd.DataFrame,
    axis: str,
    min_subfamily_events: int,
) -> dict[str, Any]:
    subfamilies = _subfamilies_for_axis(
        registry_row=registry_row,
        family_events=family_events,
        axis=axis,
        min_subfamily_events=min_subfamily_events,
    )
    eligible = subfamilies[subfamilies["event_count"].ge(min_subfamily_events)]
    recovered = eligible[
        eligible["definition_refinement_result"].eq(
            "recovered_coherent_endpoint_vector_subfamily"
        )
    ]
    numeric = eligible[
        eligible["definition_refinement_result"].eq("recovered_numeric_stress_subfamily")
    ]
    tiny = subfamilies[subfamilies["event_count"].lt(min_subfamily_events)]
    recovered_events = int(recovered["event_count"].sum())
    numeric_events = int(numeric["event_count"].sum())
    return {
        "family_id": registry_row["family_id"],
        "boundary_family_tier": registry_row["boundary_family_tier"],
        "family_vector_class": registry_row["family_vector_class"],
        "definition_core_v1_status": registry_row["definition_core_v1_status"],
        "axis": axis,
        "axis_columns": _axis_key_label(axis),
        "is_primary_axis": axis
        == STATUS_TO_PRIMARY_AXIS[str(registry_row["definition_core_v1_status"])],
        "source_event_count": int(registry_row["event_count"]),
        "subfamily_count": int(len(subfamilies)),
        "eligible_subfamily_count": int(len(eligible)),
        "tiny_subfamily_count": int(len(tiny)),
        "tiny_event_count": int(tiny["event_count"].sum()) if not tiny.empty else 0,
        "recovered_coherent_subfamily_count": int(len(recovered)),
        "recovered_coherent_event_count": recovered_events,
        "recovered_coherent_event_share": recovered_events / float(registry_row["event_count"]),
        "recovered_numeric_stress_subfamily_count": int(len(numeric)),
        "recovered_numeric_stress_event_count": numeric_events,
        "recovered_numeric_stress_event_share": numeric_events / float(registry_row["event_count"]),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _family_refinement_read(row: pd.Series) -> str:
    coverage = float(row["primary_recovered_coherent_event_share"])
    numeric_coverage = float(row["primary_recovered_numeric_stress_event_share"])
    if coverage >= 0.75:
        return "primary_decomposition_recovers_most_events"
    if coverage + numeric_coverage >= 0.75:
        return "primary_decomposition_recovers_most_events_with_numeric_stress"
    if coverage > 0.0:
        return "primary_decomposition_recovers_partial_events"
    if float(row["best_axis_recovered_coherent_event_share"]) >= 0.75:
        return "alternative_axis_recovers_most_events"
    if float(row["best_axis_recovered_coherent_event_share"]) > 0.0:
        return "alternative_axis_recovers_partial_events"
    return "no_coherent_recovery_under_current_axes"


def _build_outputs(
    *,
    registry: pd.DataFrame,
    events: pd.DataFrame,
    min_subfamily_events: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    queue = registry[registry["definition_core_v1_status"].ne(ACCEPTED_STATUS)].copy()
    event_rows: list[pd.DataFrame] = []
    primary_subfamilies: list[pd.DataFrame] = []
    axis_rows: list[dict[str, Any]] = []

    for registry_row in queue.itertuples(index=False):
        row = pd.Series(registry_row._asdict())
        family_events = events[events["family_id"].eq(row["family_id"])].copy()
        if family_events.empty:
            raise ValueError(f"missing events for queue family: {row['family_id']}")
        primary_axis = STATUS_TO_PRIMARY_AXIS[str(row["definition_core_v1_status"])]
        primary = _subfamilies_for_axis(
            registry_row=row,
            family_events=family_events,
            axis=primary_axis,
            min_subfamily_events=min_subfamily_events,
        )
        primary_subfamilies.append(primary)
        event_family = family_events.copy()
        event_family["decomposition_axis"] = primary_axis
        event_family["decomposition_axis_columns"] = _axis_key_label(primary_axis)
        event_family["decomposition_key"] = event_family.apply(
            lambda event: _axis_key(event, primary_axis),
            axis=1,
        )
        event_family = event_family.merge(
            primary[
                [
                    "source_family_id",
                    "subfamily_id",
                    "decomposition_key",
                    "subfamily_coherence_status",
                    "definition_refinement_result",
                ]
            ],
            left_on=["family_id", "decomposition_key"],
            right_on=["source_family_id", "decomposition_key"],
            how="left",
            validate="many_to_one",
        )
        event_rows.append(event_family)
        for axis in AXIS_COLUMNS:
            axis_rows.append(
                _axis_comparison_row(
                    registry_row=row,
                    family_events=family_events,
                    axis=axis,
                    min_subfamily_events=min_subfamily_events,
                )
            )

    subfamily_rows = (
        pd.concat(primary_subfamilies, ignore_index=True, sort=False)
        if primary_subfamilies
        else pd.DataFrame()
    )
    subfamily_event_rows = (
        pd.concat(event_rows, ignore_index=True, sort=False) if event_rows else pd.DataFrame()
    )
    axis_comparison = pd.DataFrame(axis_rows)
    primary_axis = axis_comparison[axis_comparison["is_primary_axis"]].copy()
    best_axis = (
        axis_comparison.sort_values(
            [
                "family_id",
                "recovered_coherent_event_share",
                "recovered_coherent_subfamily_count",
                "tiny_event_count",
            ],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("family_id")
        .rename(
            columns={
                "axis": "best_axis",
                "axis_columns": "best_axis_columns",
                "recovered_coherent_subfamily_count": "best_axis_recovered_coherent_subfamily_count",
                "recovered_coherent_event_count": "best_axis_recovered_coherent_event_count",
                "recovered_coherent_event_share": "best_axis_recovered_coherent_event_share",
            }
        )
    )
    family_decomposition = primary_axis.rename(
        columns={
            "axis": "primary_axis",
            "axis_columns": "primary_axis_columns",
            "subfamily_count": "primary_subfamily_count",
            "eligible_subfamily_count": "primary_eligible_subfamily_count",
            "tiny_subfamily_count": "primary_tiny_subfamily_count",
            "tiny_event_count": "primary_tiny_event_count",
            "recovered_coherent_subfamily_count": "primary_recovered_coherent_subfamily_count",
            "recovered_coherent_event_count": "primary_recovered_coherent_event_count",
            "recovered_coherent_event_share": "primary_recovered_coherent_event_share",
            "recovered_numeric_stress_subfamily_count": "primary_recovered_numeric_stress_subfamily_count",
            "recovered_numeric_stress_event_count": "primary_recovered_numeric_stress_event_count",
            "recovered_numeric_stress_event_share": "primary_recovered_numeric_stress_event_share",
        }
    ).merge(
        best_axis[
            [
                "family_id",
                "best_axis",
                "best_axis_columns",
                "best_axis_recovered_coherent_subfamily_count",
                "best_axis_recovered_coherent_event_count",
                "best_axis_recovered_coherent_event_share",
            ]
        ],
        on="family_id",
        how="left",
        validate="one_to_one",
    )
    family_decomposition["family_refinement_read"] = family_decomposition.apply(
        _family_refinement_read,
        axis=1,
    )
    family_decomposition["claim_boundary"] = CLAIM_BOUNDARY
    class_summary = (
        subfamily_rows.groupby(
            [
                "source_definition_core_v1_status",
                "decomposition_axis",
                "definition_refinement_result",
                "subfamily_coherence_status",
            ],
            as_index=False,
        )
        .agg(
            subfamily_count=("subfamily_id", "size"),
            source_family_count=("source_family_id", "nunique"),
            event_count_sum=("event_count", "sum"),
            median_event_count=("event_count", "median"),
            median_split_class_share=("dominant_split_vector_class_share", "median"),
            median_host_context_share=("dominant_host_context_class_share", "median"),
            median_shape_core_share=("dominant_shape_core_signature_share", "median"),
            median_host_handle_share=("dominant_host_handle_share", "median"),
        )
        .sort_values(
            ["source_definition_core_v1_status", "definition_refinement_result", "event_count_sum"],
            ascending=[True, True, False],
        )
    )
    class_summary["claim_boundary"] = CLAIM_BOUNDARY
    recovered_shortlist = subfamily_rows[
        subfamily_rows["definition_refinement_result"].isin(
            {
                "recovered_coherent_endpoint_vector_subfamily",
                "recovered_numeric_stress_subfamily",
            }
        )
        & subfamily_rows["event_count"].ge(min_subfamily_events)
    ].copy()
    recovered_shortlist["definition_recovery_rank_score"] = (
        recovered_shortlist["dominant_split_vector_class_share"].astype(float)
        + recovered_shortlist["dominant_host_context_class_share"].astype(float)
        + recovered_shortlist["dominant_shape_core_signature_share"].astype(float)
        + recovered_shortlist["dominant_host_handle_share"].astype(float)
        + recovered_shortlist["event_count_share_of_source_family"].astype(float)
        - recovered_shortlist["top1_segment_share_iqr"].astype(float)
        - recovered_shortlist["top2_segment_share_iqr"].astype(float)
    )
    recovered_shortlist = recovered_shortlist.sort_values(
        [
            "definition_refinement_result",
            "boundary_family_tier",
            "source_family_vector_class",
            "definition_recovery_rank_score",
            "event_count",
            "source_ref_weight_sum",
        ],
        ascending=[True, True, True, False, False, False],
    )
    return (
        subfamily_event_rows,
        subfamily_rows,
        family_decomposition,
        axis_comparison,
        class_summary,
        recovered_shortlist,
    )


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
    suffix: list[str] = []
    if len(frame) > max_rows:
        suffix.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    subfamily_rows: pd.DataFrame,
    family_decomposition: pd.DataFrame,
    axis_comparison: pd.DataFrame,
    class_summary: pd.DataFrame,
    recovered_shortlist: pd.DataFrame,
    min_subfamily_events: int,
) -> None:
    recovered = subfamily_rows[
        subfamily_rows["definition_refinement_result"].eq(
            "recovered_coherent_endpoint_vector_subfamily"
        )
    ]
    family_rollup = (
        family_decomposition.groupby(
            ["definition_core_v1_status", "family_refinement_read"],
            as_index=False,
        )
        .agg(
            family_count=("family_id", "size"),
            source_event_count_sum=("source_event_count", "sum"),
            primary_recovered_event_count_sum=(
                "primary_recovered_coherent_event_count",
                "sum",
            ),
            median_primary_recovered_event_share=(
                "primary_recovered_coherent_event_share",
                "median",
            ),
        )
        .sort_values(["definition_core_v1_status", "family_count"], ascending=[True, False])
    )
    axis_rollup = (
        axis_comparison.groupby(["definition_core_v1_status", "axis"], as_index=False)
        .agg(
            family_count=("family_id", "size"),
            recovered_event_count_sum=("recovered_coherent_event_count", "sum"),
            median_recovered_event_share=("recovered_coherent_event_share", "median"),
            median_tiny_event_count=("tiny_event_count", "median"),
        )
        .sort_values(["definition_core_v1_status", "recovered_event_count_sum"], ascending=[True, False])
    )
    text = [
        "# NanoClustering Definition-Core V1 Refinement Queue Decomposition",
        "",
        f"- subfamily_rows: `{len(subfamily_rows)}`",
        f"- source_queue_families: `{family_decomposition['family_id'].nunique()}`",
        f"- recovered_coherent_subfamilies: `{len(recovered)}`",
        f"- recovered_shortlist_rows: `{len(recovered_shortlist)}`",
        f"- min_subfamily_events: `{min_subfamily_events}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Family Recovery Rollup",
        "",
        _markdown_table(
            family_rollup,
            [
                "definition_core_v1_status",
                "family_refinement_read",
                "family_count",
                "source_event_count_sum",
                "primary_recovered_event_count_sum",
                "median_primary_recovered_event_share",
            ],
            max_rows=30,
        ),
        "",
        "## Axis Recovery Rollup",
        "",
        _markdown_table(
            axis_rollup,
            [
                "definition_core_v1_status",
                "axis",
                "family_count",
                "recovered_event_count_sum",
                "median_recovered_event_share",
                "median_tiny_event_count",
            ],
            max_rows=40,
        ),
        "",
        "## Subfamily Class Summary",
        "",
        _markdown_table(
            class_summary,
            [
                "source_definition_core_v1_status",
                "decomposition_axis",
                "definition_refinement_result",
                "subfamily_coherence_status",
                "subfamily_count",
                "source_family_count",
                "event_count_sum",
                "median_event_count",
            ],
            max_rows=40,
        ),
        "",
        "## Recovered Subfamily Shortlist",
        "",
        _markdown_table(
            recovered_shortlist,
            [
                "subfamily_id",
                "source_family_id",
                "source_definition_core_v1_status",
                "source_family_vector_class",
                "subfamily_vector_class",
                "event_count",
                "event_count_share_of_source_family",
                "dominant_split_vector_class_share",
                "dominant_host_context_class_share",
                "dominant_shape_core_signature_share",
                "dominant_host_handle_share",
            ],
            max_rows=30,
        ),
        "",
        "## Read",
        "",
        "- This is a definition-refinement diagnostic, not a wall/pathway candidate promotion.",
        "- Recovered coherent subfamilies show where the v1 family unit was too coarse.",
        "- Tiny subfamilies are recorded separately so singleton partitions do not inflate the accepted basin count.",
        "- The current evidence favors split-vector class as the first split for split-mixed and heterogeneous queues, and host-context class as the first split for host-variable queues; shape-core signatures remain a secondary stability check.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    registry_dir: Path,
    coherence_dir: Path,
    output_dir: Path,
    min_subfamily_events: int,
) -> dict[str, Any]:
    registry = _read_csv(registry_dir / V1_FAMILY_REGISTRY_CSV)
    events = _read_csv(coherence_dir / COHERENCE_EVENT_ROWS_CSV)
    (
        subfamily_event_rows,
        subfamily_rows,
        family_decomposition,
        axis_comparison,
        class_summary,
        recovered_shortlist,
    ) = _build_outputs(
        registry=registry,
        events=events,
        min_subfamily_events=min_subfamily_events,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(subfamily_event_rows, output_dir / SUBFAMILY_EVENT_ROWS_CSV)
    _write_csv(subfamily_rows, output_dir / SUBFAMILY_ROWS_CSV)
    _write_csv(family_decomposition, output_dir / FAMILY_DECOMPOSITION_ROWS_CSV)
    _write_csv(axis_comparison, output_dir / AXIS_COMPARISON_ROWS_CSV)
    _write_csv(class_summary, output_dir / CLASS_SUMMARY_CSV)
    _write_csv(recovered_shortlist, output_dir / RECOVERED_SHORTLIST_CSV)
    _write_report(
        output_dir=output_dir,
        subfamily_rows=subfamily_rows,
        family_decomposition=family_decomposition,
        axis_comparison=axis_comparison,
        class_summary=class_summary,
        recovered_shortlist=recovered_shortlist,
        min_subfamily_events=min_subfamily_events,
    )

    recovered = subfamily_rows[
        subfamily_rows["definition_refinement_result"].eq(
            "recovered_coherent_endpoint_vector_subfamily"
        )
    ]
    tiny = subfamily_rows[
        subfamily_rows["definition_refinement_result"].eq("diagnostic_tiny_subfamily_not_promoted")
    ]
    summary = {
        "ok": True,
        "registry_dir": _rel(registry_dir),
        "coherence_dir": _rel(coherence_dir),
        "output_dir": _rel(output_dir),
        "source_queue_family_count": int(family_decomposition["family_id"].nunique()),
        "source_queue_event_count": int(family_decomposition["source_event_count"].sum()),
        "subfamily_row_count": int(len(subfamily_rows)),
        "subfamily_event_row_count": int(len(subfamily_event_rows)),
        "axis_comparison_row_count": int(len(axis_comparison)),
        "recovered_coherent_subfamily_count": int(len(recovered)),
        "recovered_coherent_source_family_count": int(recovered["source_family_id"].nunique()),
        "recovered_coherent_event_count": int(recovered["event_count"].sum()),
        "tiny_subfamily_count": int(len(tiny)),
        "tiny_event_count": int(tiny["event_count"].sum()) if not tiny.empty else 0,
        "family_refinement_read_counts": _count(family_decomposition, "family_refinement_read"),
        "definition_refinement_result_counts": _count(
            subfamily_rows,
            "definition_refinement_result",
        ),
        "source_status_counts": _count(family_decomposition, "definition_core_v1_status"),
        "min_subfamily_events": min_subfamily_events,
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "subfamily_event_rows_csv": _rel(output_dir / SUBFAMILY_EVENT_ROWS_CSV),
            "subfamily_rows_csv": _rel(output_dir / SUBFAMILY_ROWS_CSV),
            "family_decomposition_rows_csv": _rel(output_dir / FAMILY_DECOMPOSITION_ROWS_CSV),
            "axis_comparison_rows_csv": _rel(output_dir / AXIS_COMPARISON_ROWS_CSV),
            "class_summary_csv": _rel(output_dir / CLASS_SUMMARY_CSV),
            "recovered_shortlist_csv": _rel(output_dir / RECOVERED_SHORTLIST_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "registry_dir": _rel(registry_dir),
        "coherence_dir": _rel(coherence_dir),
        "output_dir": _rel(output_dir),
        "accepted_status_excluded_from_queue": ACCEPTED_STATUS,
        "status_to_primary_axis": STATUS_TO_PRIMARY_AXIS,
        "axis_columns": AXIS_COLUMNS,
        "min_subfamily_events": min_subfamily_events,
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
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--coherence-dir", type=Path, default=DEFAULT_COHERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-subfamily-events",
        type=int,
        default=2,
        help="Minimum repeated events required before a decomposed subfamily can count as recovered.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        registry_dir=args.registry_dir.resolve(),
        coherence_dir=args.coherence_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        min_subfamily_events=args.min_subfamily_events,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
