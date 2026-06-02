#!/usr/bin/env python3
"""Materialize the v2.2 accepted-primitive measurement panel.

This reads the frozen NanoClustering v2.2 basin-definition surface and expands
accepted primitives into measurement rows. It measures recurrence support,
endpoint-vector composition, host-handle concentration, and residual-debt
caveats while keeping stress/control strata outside the accepted panel.

It does not run clustering, execute optimizer routes, promote wall/pathway
claims, or inspect basin quality/cost.
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
DEFAULT_V2_2_REGISTRY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_2_exception_axis_registry_20260531"
)
DEFAULT_INSTRUMENTATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_v2_2_instrumentation_surface_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_v2_2_measurement_panel_20260531"
)

V2_2_PRIMITIVE_REGISTRY_CSV = (
    "nanoclustering_definition_core_v2_2_primitive_registry.csv"
)
V2_2_PRIMITIVE_EVENT_ROWS_CSV = (
    "nanoclustering_definition_core_v2_2_primitive_event_rows.csv"
)
V2_2_RESIDUAL_DEFINITION_QUEUE_CSV = (
    "nanoclustering_definition_core_v2_2_residual_definition_queue.csv"
)
FAMILY_INSTRUMENTATION_ROWS_CSV = (
    "nanoclustering_v2_2_family_instrumentation_rows.csv"
)

ACCEPTED_PRIMITIVE_MEASUREMENT_ROWS_CSV = (
    "nanoclustering_v2_2_accepted_primitive_measurement_rows.csv"
)
ACCEPTED_PRIMITIVE_EVENT_MEASUREMENT_ROWS_CSV = (
    "nanoclustering_v2_2_accepted_primitive_event_measurement_rows.csv"
)
SOURCE_FAMILY_MEASUREMENT_ROLLUP_CSV = (
    "nanoclustering_v2_2_source_family_measurement_rollup.csv"
)
MEASUREMENT_SUPPORT_SUMMARY_CSV = (
    "nanoclustering_v2_2_measurement_support_summary.csv"
)
MEASUREMENT_GATE_MATRIX_CSV = "nanoclustering_v2_2_measurement_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_v2_2_measurement_summary.json"
REPORT_MD = "nanoclustering_v2_2_measurement_report.md"
CONFIG_JSON = "nanoclustering_v2_2_measurement_config.json"

CLAIM_BOUNDARY = (
    "V2.2 accepted-primitive measurement panel only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search "
    "claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_v2_2_measurement_panel"

CRITICAL_EVENT_COLUMNS = [
    "split_vector_class",
    "host_context_class",
    "shape_core_signature",
    "boundary_pattern",
    "dominant_host_handle_id",
    "top1_segment_share_ref_weight",
    "effective_segment_count",
]
NUMERIC_EVENT_COLUMNS = [
    "top1_segment_share_ref_weight",
    "top2_segment_share_ref_weight",
    "top3_segment_share_ref_weight",
    "top2_segment_share_sum",
    "top3_segment_share_sum",
    "effective_segment_count",
    "target_share_of_best_run_cluster_weight",
    "dominant_host_share_run_weight",
    "dominant_host_share_own_ref_weight",
    "fragmentation_index",
    "split_segment_count_ge5_weight",
    "merge_contributor_count_ge5_weight",
    "significant_segment_count",
    "ref_weight_sum",
]
BOOL_EVENT_COLUMNS = [
    "dominant_host_is_source_ref",
    "is_strong_boundary_seed",
    "is_severe_boundary_seed",
    "is_strong_fragmentation_event",
    "is_severe_fragmentation_event",
    "is_moderate_fragmentation_event",
]


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
        return [_json_safe(item)
                for item in value]
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


def _joined_unique(values: pd.Series) -> str:
    clean = sorted({str(value) for value in values.dropna() if str(value)})
    return ";".join(clean)


def _joined_seed_list(values: pd.Series) -> str:
    clean = sorted({str(int(value)) if pd.notna(value) else "" for value in values})
    return ",".join([value for value in clean if value])


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = frame.copy()
    for column in columns:
        if column in rows:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def _booleanize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = frame.copy()
    truthy = {"true", "1", "yes", "y"}
    for column in columns:
        if column not in rows:
            continue
        rows[column] = rows[column].map(
            lambda value: str(value).strip().lower() in truthy if pd.notna(value) else False
        )
    return rows


def _mode_stats(values: pd.Series) -> tuple[str, int, float, str]:
    clean = [str(value) for value in values.dropna() if str(value)]
    if not clean:
        return "", 0, 0.0, ""
    counts = pd.Series(clean).value_counts().sort_index()
    max_count = int(counts.max())
    mode_value = sorted([key for key, value in counts.items() if int(value) == max_count])[0]
    share = max_count / len(clean) if clean else 0.0
    counts_str = ";".join(
        f"{key}:{int(value)}"
        for key, value in counts.sort_values(ascending=False).items()
    )
    return mode_value, max_count, share, counts_str


def _support_measurement_class(event_count: int) -> str:
    if event_count >= 5:
        return "deep_support_measurement_unit"
    if event_count >= 3:
        return "moderate_support_measurement_unit"
    if event_count >= 2:
        return "thin_support_measurement_unit"
    return "singleton_support_not_expected_in_accepted_panel"


def _residual_caveat_status(residual_events: int) -> str:
    if residual_events > 0:
        return "source_family_has_residual_definition_debt"
    return "source_family_has_no_residual_definition_debt"


def _endpoint_host_scope(source_ref_share: float) -> str:
    if source_ref_share >= 0.999:
        return "all_events_source_host_preserved"
    if source_ref_share <= 0.001:
        return "all_events_external_host_absorption"
    return "mixed_source_and_external_host_events"


def _event_stats(event_rows: pd.DataFrame) -> pd.DataFrame:
    rows = _numeric(event_rows, NUMERIC_EVENT_COLUMNS + ["comparison_seed"])
    rows = _booleanize(rows, BOOL_EVENT_COLUMNS)
    grouped_rows: list[dict[str, Any]] = []
    for primitive_id, group in rows.groupby("primitive_id", sort=True):
        row: dict[str, Any] = {
            "primitive_id": primitive_id,
            "event_count_from_rows": int(group["event_id"].nunique()),
            "distinct_comparison_seed_count": int(group["comparison_seed"].nunique()),
            "comparison_seed_list": _joined_seed_list(group["comparison_seed"]),
        }
        for column in [
            "split_vector_class",
            "host_context_class",
            "shape_core_signature",
            "boundary_pattern",
            "dominant_host_handle_id",
            "top1_endpoint_handle_id",
        ]:
            mode_value, mode_count, mode_share, counts_str = _mode_stats(group[column])
            row[f"{column}_mode"] = mode_value
            row[f"{column}_mode_count"] = mode_count
            row[f"{column}_mode_share"] = mode_share
            row[f"{column}_counts"] = counts_str
            row[f"{column}_distinct_count"] = int(group[column].dropna().astype(str).nunique())
        for column in NUMERIC_EVENT_COLUMNS:
            if column not in group:
                continue
            series = group[column].dropna()
            row[f"{column}_median"] = float(series.median()) if not series.empty else math.nan
            row[f"{column}_min"] = float(series.min()) if not series.empty else math.nan
            row[f"{column}_max"] = float(series.max()) if not series.empty else math.nan
        for column in BOOL_EVENT_COLUMNS:
            if column not in group:
                continue
            row[f"{column}_count"] = int(group[column].sum())
            row[f"{column}_share"] = float(group[column].mean()) if len(group) else 0.0
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows)


def _family_lookup(family_rows: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "family_id",
        "v2_2_family_status",
        "instrumentation_role",
        "accepted_primitive_count",
        "accepted_event_count",
        "residual_row_count",
        "residual_event_count",
        "residual_queue_statuses",
        "ref_unit_count",
        "ref_weight_sum",
        "comparison_seed_count",
        "strong_seed_count",
        "severe_seed_count",
        "moderate_seed_count",
        "top_split_share_min",
        "top_split_share_median",
        "fragmentation_index_median",
    ]
    lookup = family_rows[[column for column in keep if column in family_rows]].copy()
    return lookup.rename(
        columns={
            "family_id": "source_family_id",
            "accepted_primitive_count": "source_family_accepted_primitive_count",
            "accepted_event_count": "source_family_accepted_event_count",
            "residual_row_count": "source_family_residual_row_count",
            "residual_event_count": "source_family_residual_event_count",
            "ref_unit_count": "source_family_ref_unit_count",
            "ref_weight_sum": "source_family_ref_weight_sum",
            "comparison_seed_count": "source_family_comparison_seed_count",
            "strong_seed_count": "source_family_strong_seed_count",
            "severe_seed_count": "source_family_severe_seed_count",
            "moderate_seed_count": "source_family_moderate_seed_count",
            "top_split_share_min": "source_family_top_split_share_min",
            "top_split_share_median": "source_family_top_split_share_median",
            "fragmentation_index_median": "source_family_fragmentation_index_median",
        }
    )


def _primitive_measurement_rows(
    *,
    registry: pd.DataFrame,
    event_rows: pd.DataFrame,
    family_rows: pd.DataFrame,
) -> pd.DataFrame:
    registry = _numeric(
        registry,
        [
            "event_count",
            "source_event_count",
            "event_count_share_of_source_family",
            "dominant_split_vector_class_share",
            "dominant_host_context_class_share",
            "dominant_shape_core_signature_share",
            "dominant_host_handle_share",
            "top1_segment_share_median",
            "top2_segment_share_median",
            "effective_segment_count_median",
        ],
    )
    rows = registry.merge(_event_stats(event_rows), on="primitive_id", how="left", validate="one_to_one")
    rows = rows.merge(_family_lookup(family_rows), on="source_family_id", how="left", validate="many_to_one")
    if "effective_segment_count_median_y" in rows:
        rows["effective_segment_count_median"] = rows["effective_segment_count_median_y"]
    elif "effective_segment_count_median_x" in rows:
        rows["effective_segment_count_median"] = rows["effective_segment_count_median_x"]
    for column in [
        "source_family_residual_row_count",
        "source_family_residual_event_count",
        "source_family_accepted_primitive_count",
        "source_family_accepted_event_count",
    ]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0).astype(int)
    rows["event_count_mismatch"] = rows["event_count"].astype(int) - rows[
        "event_count_from_rows"
    ].astype(int)
    rows["measurement_support_class"] = rows["event_count"].astype(int).map(
        _support_measurement_class
    )
    rows["residual_caveat_status"] = rows["source_family_residual_event_count"].map(
        _residual_caveat_status
    )
    rows["endpoint_host_scope"] = rows["dominant_host_is_source_ref_share"].map(
        _endpoint_host_scope
    )
    rows["measurement_panel_role"] = "accepted_v2_2_primitive_measurement_unit"
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "primitive_id",
        "measurement_panel_role",
        "source_family_id",
        "branch",
        "ref_cluster_id",
        "boundary_family_tier",
        "primitive_type",
        "definition_core_v2_2_rule_status",
        "definition_confidence_tier",
        "support_depth_tier",
        "measurement_support_class",
        "event_count",
        "event_count_from_rows",
        "event_count_mismatch",
        "distinct_comparison_seed_count",
        "comparison_seed_list",
        "source_event_count",
        "event_count_share_of_source_family",
        "source_family_accepted_primitive_count",
        "source_family_accepted_event_count",
        "source_family_residual_row_count",
        "source_family_residual_event_count",
        "residual_queue_statuses",
        "residual_caveat_status",
        "split_vector_class_mode",
        "split_vector_class_mode_share",
        "host_context_class_mode",
        "host_context_class_mode_share",
        "shape_core_signature_mode",
        "shape_core_signature_mode_share",
        "boundary_pattern_mode",
        "boundary_pattern_mode_share",
        "dominant_host_handle_id_mode",
        "dominant_host_handle_id_mode_share",
        "top1_endpoint_handle_id_distinct_count",
        "top1_endpoint_handle_id_mode_share",
        "endpoint_host_scope",
        "top1_segment_share_ref_weight_median",
        "top1_segment_share_ref_weight_min",
        "top1_segment_share_ref_weight_max",
        "effective_segment_count_median",
        "target_share_of_best_run_cluster_weight_median",
        "fragmentation_index_median",
        "split_segment_count_ge5_weight_median",
        "merge_contributor_count_ge5_weight_median",
        "is_strong_boundary_seed_count",
        "is_severe_boundary_seed_count",
        "is_strong_fragmentation_event_count",
        "is_severe_fragmentation_event_count",
        "is_moderate_fragmentation_event_count",
        "source_family_comparison_seed_count",
        "source_family_strong_seed_count",
        "source_family_severe_seed_count",
        "source_family_moderate_seed_count",
        "source_family_top_split_share_median",
        "source_family_fragmentation_index_median",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    return rows[[column for column in preferred if column in rows.columns]].sort_values(
        ["boundary_family_tier", "branch", "source_family_id", "primitive_id"]
    )


def _event_measurement_rows(
    *,
    event_rows: pd.DataFrame,
    primitive_rows: pd.DataFrame,
) -> pd.DataFrame:
    lookup = primitive_rows[
        [
            "primitive_id",
            "measurement_support_class",
            "residual_caveat_status",
            "source_family_residual_event_count",
        ]
    ]
    rows = event_rows.merge(lookup, on="primitive_id", how="left", validate="many_to_one")
    rows["measurement_panel_role"] = "accepted_v2_2_primitive_event_measurement_row"
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    keep = [
        "primitive_id",
        "measurement_panel_role",
        "source_family_id",
        "event_id",
        "branch",
        "boundary_family_tier",
        "comparison_seed",
        "measurement_support_class",
        "residual_caveat_status",
        "source_family_residual_event_count",
        "split_vector_class",
        "host_context_class",
        "shape_core_signature",
        "boundary_pattern",
        "dominant_host_handle_id",
        "dominant_host_is_source_ref",
        "top1_endpoint_handle_id",
        "top1_segment_share_ref_weight",
        "top2_segment_share_ref_weight",
        "effective_segment_count",
        "target_share_of_best_run_cluster_weight",
        "fragmentation_index",
        "split_segment_count_ge5_weight",
        "merge_contributor_count_ge5_weight",
        "is_strong_boundary_seed",
        "is_severe_boundary_seed",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    return rows[[column for column in keep if column in rows.columns]].sort_values(
        ["boundary_family_tier", "branch", "source_family_id", "primitive_id", "event_id"]
    )


def _source_family_rollup(primitive_rows: pd.DataFrame) -> pd.DataFrame:
    rows = (
        primitive_rows.groupby("source_family_id", as_index=False)
        .agg(
            branch=("branch", "first"),
            ref_cluster_id=("ref_cluster_id", "first"),
            boundary_family_tier=("boundary_family_tier", "first"),
            primitive_count=("primitive_id", "nunique"),
            accepted_event_count=("event_count", "sum"),
            residual_event_count=("source_family_residual_event_count", "first"),
            residual_row_count=("source_family_residual_row_count", "first"),
            residual_queue_statuses=("residual_queue_statuses", "first"),
            measurement_support_classes=("measurement_support_class", _joined_unique),
            definition_confidence_tiers=("definition_confidence_tier", _joined_unique),
            primitive_types=("primitive_type", _joined_unique),
            split_vector_modes=("split_vector_class_mode", _joined_unique),
            host_context_modes=("host_context_class_mode", _joined_unique),
            boundary_pattern_modes=("boundary_pattern_mode", _joined_unique),
            dominant_host_handle_count=("dominant_host_handle_id_mode", "nunique"),
            median_host_mode_share=("dominant_host_handle_id_mode_share", "median"),
            median_shape_mode_share=("shape_core_signature_mode_share", "median"),
            median_top1_segment_share=("top1_segment_share_ref_weight_median", "median"),
            median_effective_segment_count=("effective_segment_count_median", "median"),
            median_fragmentation_index=("fragmentation_index_median", "median"),
        )
        .sort_values(["boundary_family_tier", "branch", "source_family_id"])
    )
    rows["residual_caveat_status"] = rows["residual_event_count"].map(_residual_caveat_status)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _support_summary(primitive_rows: pd.DataFrame) -> pd.DataFrame:
    rows = _numeric(
        primitive_rows,
        [
            "event_count",
            "source_family_residual_event_count",
            "top1_segment_share_ref_weight_median",
            "effective_segment_count_median",
            "dominant_host_handle_id_mode_share",
            "shape_core_signature_mode_share",
            "fragmentation_index_median",
        ],
    )
    group_cols = [
        "boundary_family_tier",
        "primitive_type",
        "definition_confidence_tier",
        "measurement_support_class",
        "residual_caveat_status",
        "host_context_class_mode",
        "boundary_pattern_mode",
    ]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in rows.groupby(group_cols, dropna=False, sort=True):
        row = {column: value for column, value in zip(group_cols, keys)}
        residual_by_family = group[["source_family_id", "source_family_residual_event_count"]].drop_duplicates(
            "source_family_id"
        )
        row.update(
            {
                "primitive_count": int(group["primitive_id"].nunique()),
                "source_family_count": int(group["source_family_id"].nunique()),
                "event_count": int(group["event_count"].sum()),
                "source_family_with_residual_count": int(
                    residual_by_family["source_family_residual_event_count"].gt(0).sum()
                ),
                "distinct_family_residual_event_count_sum": int(
                    residual_by_family["source_family_residual_event_count"].sum()
                ),
                "median_top1_segment_share": float(
                    group["top1_segment_share_ref_weight_median"].median()
                ),
                "median_effective_segment_count": float(
                    group["effective_segment_count_median"].median()
                ),
                "median_host_handle_mode_share": float(
                    group["dominant_host_handle_id_mode_share"].median()
                ),
                "median_shape_mode_share": float(
                    group["shape_core_signature_mode_share"].median()
                ),
                "median_fragmentation_index": float(group["fragmentation_index_median"].median()),
            }
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "boundary_family_tier",
            "primitive_type",
            "definition_confidence_tier",
            "measurement_support_class",
            "residual_caveat_status",
        ]
    )
    summary["route_execution_status"] = ROUTE_EXECUTION_STATUS
    summary["wall_promotion_status"] = WALL_PROMOTION_STATUS
    summary["quality_cost_status"] = QUALITY_COST_STATUS
    summary["claim_boundary"] = CLAIM_BOUNDARY
    return summary


def _gate_matrix(
    *,
    registry: pd.DataFrame,
    event_rows: pd.DataFrame,
    residual_queue: pd.DataFrame,
    primitive_rows: pd.DataFrame,
    family_rollup: pd.DataFrame,
) -> pd.DataFrame:
    primitive_row_count = int(registry["primitive_id"].nunique())
    primitive_event_count = int(registry["event_count"].sum())
    event_row_count = int(event_rows["event_id"].nunique())
    duplicate_event_rows = int(event_rows.duplicated(["primitive_id", "event_id"]).sum())
    mismatch_count = int(primitive_rows["event_count_mismatch"].ne(0).sum())
    residual_events = int(pd.to_numeric(residual_queue["event_count"], errors="coerce").sum())
    family_rollup_event_count = int(family_rollup["accepted_event_count"].sum())
    accepted_family_count = int(family_rollup["source_family_id"].nunique())
    families_with_residual = int(family_rollup["residual_event_count"].gt(0).sum())
    critical_missing = int(
        event_rows[CRITICAL_EVENT_COLUMNS].isna().sum().sum()
    )
    singleton_accepted = int(
        primitive_rows["measurement_support_class"].eq(
            "singleton_support_not_expected_in_accepted_panel"
        ).sum()
    )
    route_status_ok = bool(
        primitive_rows["route_execution_status"].eq(ROUTE_EXECUTION_STATUS).all()
        and primitive_rows["wall_promotion_status"].eq(WALL_PROMOTION_STATUS).all()
        and primitive_rows["quality_cost_status"].eq(QUALITY_COST_STATUS).all()
    )
    rows = [
        {
            "gate_id": "M1_accepted_primitive_accounting",
            "gate_question": "Does the accepted primitive panel preserve the frozen v2.2 event accounting?",
            "evidence": (
                f"primitive_rows={primitive_row_count}, registry_events={primitive_event_count}, "
                f"event_rows={event_row_count}, family_rollup_events={family_rollup_event_count}, "
                f"duplicate_event_rows={duplicate_event_rows}, registry_event_mismatches={mismatch_count}"
            ),
            "status": (
                "pass"
                if primitive_row_count == 223
                and primitive_event_count == 910
                and event_row_count == 910
                and family_rollup_event_count == 910
                and duplicate_event_rows == 0
                and mismatch_count == 0
                else "blocked"
            ),
            "decision": "accepted_primitive_measurement_rows_are_accountable",
            "next_action": "use primitive rows as the measurement unit",
        },
        {
            "gate_id": "M2_residual_debt_caveats_attached",
            "gate_question": "Are residual-definition rows carried as caveats rather than dropped?",
            "evidence": (
                f"residual_events={residual_events}, accepted_families={accepted_family_count}, "
                f"families_with_residual_debt={families_with_residual}"
            ),
            "status": "pass" if residual_events == 116 and families_with_residual == 41 else "blocked",
            "decision": "carry residual caveats at source-family level",
            "next_action": "exclude residual-only rows from accepted measurements but report their debt",
        },
        {
            "gate_id": "M3_endpoint_vector_measurement_coverage",
            "gate_question": "Do accepted event rows have the endpoint-vector fields needed for measurement?",
            "evidence": (
                f"critical_missing_values={critical_missing}, singleton_accepted_primitives={singleton_accepted}"
            ),
            "status": "pass" if critical_missing == 0 and singleton_accepted == 0 else "blocked",
            "decision": "endpoint-vector metrics are available for accepted primitives",
            "next_action": "summarize split-vector, host-context, shape-core, and boundary-pattern repetition",
        },
        {
            "gate_id": "M4_wall_pathway_gate",
            "gate_question": "Can this panel claim wall or pathway evidence?",
            "evidence": "no route traces are executed or inspected in this measurement panel",
            "status": "closed_no_route_evidence" if route_status_ok else "blocked_status_leak",
            "decision": "do_not_promote_wall_or_pathway_claims",
            "next_action": "open only after an explicit pathway protocol is materialized",
        },
        {
            "gate_id": "M5_quality_cost_gate",
            "gate_question": "Can this panel compare basin quality or search cost?",
            "evidence": "quality and cost are excluded by construction",
            "status": "closed_excluded_by_design" if route_status_ok else "blocked_status_leak",
            "decision": "do_not_rank_basins_by_quality_or_cost_here",
            "next_action": "defer quality/cost until existence and pathway gates are separate",
        },
        {
            "gate_id": "M6_next_research_step",
            "gate_question": "What should be executed next?",
            "evidence": "accepted primitive rows now expose support depth, endpoint-vector composition, host-handle concentration, and residual caveats",
            "status": "ready_for_distribution_review",
            "decision": "review measurement distributions before pathway design",
            "next_action": "inspect support classes and residual-debt families for the first accepted-primitive claims",
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix["claim_boundary"] = CLAIM_BOUNDARY
    return matrix


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
                values.append(str(value).replace("|", r"\|"))
        body.append("| " + " | ".join(values) + " |")
    suffix: list[str] = []
    if len(frame) > max_rows:
        suffix.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    support_summary: pd.DataFrame,
    family_rollup: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    residual_families = family_rollup[family_rollup["residual_event_count"].gt(0)].sort_values(
        ["residual_event_count", "accepted_event_count"],
        ascending=[False, False],
    )
    text = [
        "# NanoClustering V2.2 Accepted-Primitive Measurement Panel",
        "",
        f"- accepted_primitive_count: `{summary['accepted_primitive_count']}`",
        f"- accepted_event_count: `{summary['accepted_event_count']}`",
        f"- accepted_source_family_count: `{summary['accepted_source_family_count']}`",
        f"- families_with_residual_debt: `{summary['families_with_residual_debt']}`",
        f"- residual_definition_event_count: `{summary['residual_definition_event_count']}`",
        f"- critical_event_field_missing_values: `{summary['critical_event_field_missing_values']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gate_matrix,
            ["gate_id", "evidence", "status", "decision", "next_action"],
            max_rows=10,
        ),
        "",
        "## Support Summary",
        "",
        _markdown_table(
            support_summary,
            [
                "boundary_family_tier",
                "primitive_type",
                "definition_confidence_tier",
                "measurement_support_class",
                "residual_caveat_status",
                "host_context_class_mode",
                "boundary_pattern_mode",
                "primitive_count",
                "source_family_count",
                "event_count",
                "median_host_handle_mode_share",
                "median_shape_mode_share",
            ],
            max_rows=30,
        ),
        "",
        "## Residual-Debt Families",
        "",
        _markdown_table(
            residual_families,
            [
                "source_family_id",
                "branch",
                "boundary_family_tier",
                "primitive_count",
                "accepted_event_count",
                "residual_event_count",
                "residual_queue_statuses",
                "measurement_support_classes",
                "host_context_modes",
                "boundary_pattern_modes",
            ],
            max_rows=25,
        ),
        "",
        "## Read",
        "",
        "- The accepted measurement panel preserves the v2.2 accounting: 223 primitives and 910 accepted events over 166 source families.",
        "- Residual debt is visible rather than silently dropped: 41 accepted source families carry residual caveats, while residual-only families stay outside this accepted panel.",
        "- All accepted event rows have the endpoint-vector fields needed for the first measurement pass.",
        "- This panel is ready for distribution review; it still does not justify wall/pathway, quality, cost, or directed-search claims.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    v2_2_registry_dir: Path,
    instrumentation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    registry = _read_csv(v2_2_registry_dir / V2_2_PRIMITIVE_REGISTRY_CSV)
    event_rows = _read_csv(v2_2_registry_dir / V2_2_PRIMITIVE_EVENT_ROWS_CSV)
    residual_queue = _read_csv(v2_2_registry_dir / V2_2_RESIDUAL_DEFINITION_QUEUE_CSV)
    family_rows = _read_csv(instrumentation_dir / FAMILY_INSTRUMENTATION_ROWS_CSV)

    primitive_rows = _primitive_measurement_rows(
        registry=registry,
        event_rows=event_rows,
        family_rows=family_rows,
    )
    event_measurement_rows = _event_measurement_rows(
        event_rows=event_rows,
        primitive_rows=primitive_rows,
    )
    family_rollup = _source_family_rollup(primitive_rows)
    support_summary = _support_summary(primitive_rows)
    gate_matrix = _gate_matrix(
        registry=registry,
        event_rows=event_rows,
        residual_queue=residual_queue,
        primitive_rows=primitive_rows,
        family_rollup=family_rollup,
    )

    critical_missing = int(event_rows[CRITICAL_EVENT_COLUMNS].isna().sum().sum())
    summary = {
        "accepted_primitive_count": int(primitive_rows["primitive_id"].nunique()),
        "accepted_event_count": int(primitive_rows["event_count"].sum()),
        "accepted_event_row_count": int(event_measurement_rows["event_id"].nunique()),
        "accepted_source_family_count": int(family_rollup["source_family_id"].nunique()),
        "families_with_residual_debt": int(family_rollup["residual_event_count"].gt(0).sum()),
        "residual_definition_event_count": int(pd.to_numeric(residual_queue["event_count"], errors="coerce").sum()),
        "critical_event_field_missing_values": critical_missing,
        "event_count_mismatch_count": int(primitive_rows["event_count_mismatch"].ne(0).sum()),
        "measurement_support_class_counts": _count(primitive_rows, "measurement_support_class"),
        "residual_caveat_status_counts": _count(primitive_rows, "residual_caveat_status"),
        "primitive_type_counts": _count(primitive_rows, "primitive_type"),
        "host_context_mode_counts": _count(primitive_rows, "host_context_class_mode"),
        "boundary_pattern_mode_counts": _count(primitive_rows, "boundary_pattern_mode"),
        "gate_status_counts": _count(gate_matrix, "status"),
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {
            "v2_2_registry_dir": _rel(v2_2_registry_dir),
            "instrumentation_dir": _rel(instrumentation_dir),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(primitive_rows, output_dir / ACCEPTED_PRIMITIVE_MEASUREMENT_ROWS_CSV)
    _write_csv(event_measurement_rows, output_dir / ACCEPTED_PRIMITIVE_EVENT_MEASUREMENT_ROWS_CSV)
    _write_csv(family_rollup, output_dir / SOURCE_FAMILY_MEASUREMENT_ROLLUP_CSV)
    _write_csv(support_summary, output_dir / MEASUREMENT_SUPPORT_SUMMARY_CSV)
    _write_csv(gate_matrix, output_dir / MEASUREMENT_GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "v2_2_registry_dir": _rel(v2_2_registry_dir),
        "instrumentation_dir": _rel(instrumentation_dir),
        "output_dir": _rel(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        support_summary=support_summary,
        family_rollup=family_rollup,
        gate_matrix=gate_matrix,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-2-registry-dir", type=Path, default=DEFAULT_V2_2_REGISTRY_DIR)
    parser.add_argument("--instrumentation-dir", type=Path, default=DEFAULT_INSTRUMENTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = materialize(
        v2_2_registry_dir=args.v2_2_registry_dir,
        instrumentation_dir=args.instrumentation_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
