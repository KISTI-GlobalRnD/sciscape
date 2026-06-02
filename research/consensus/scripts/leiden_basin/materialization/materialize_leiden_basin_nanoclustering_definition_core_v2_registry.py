#!/usr/bin/env python3
"""Materialize the NanoClustering definition-core v2 primitive registry.

This combines accepted v1 coherent families with recovered coherent
refinement-queue subfamilies. It does not run clustering, execute optimizer
routes, promote wall/pathway claims, or inspect basin quality/cost.
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
DEFAULT_V1_REGISTRY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v1_family_registry_20260530"
)
DEFAULT_REFINEMENT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v1_refinement_queue_decomposition_20260530"
)
DEFAULT_COHERENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_full_basin_vector_coherence_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_registry_20260530"
)

V1_FAMILY_REGISTRY_CSV = "nanoclustering_definition_core_v1_family_registry.csv"
COHERENCE_EVENT_ROWS_CSV = "nanoclustering_basin_vector_coherence_event_rows.csv"
SUBFAMILY_EVENT_ROWS_CSV = "nanoclustering_definition_core_v1_refinement_subfamily_event_rows.csv"
SUBFAMILY_ROWS_CSV = "nanoclustering_definition_core_v1_refinement_subfamily_rows.csv"
FAMILY_DECOMPOSITION_ROWS_CSV = (
    "nanoclustering_definition_core_v1_refinement_family_decomposition_rows.csv"
)

PRIMITIVE_REGISTRY_CSV = "nanoclustering_definition_core_v2_primitive_registry.csv"
PRIMITIVE_EVENT_ROWS_CSV = "nanoclustering_definition_core_v2_primitive_event_rows.csv"
AUDIT_QUEUE_ROWS_CSV = "nanoclustering_definition_core_v2_audit_queue_rows.csv"
STATUS_SUMMARY_CSV = "nanoclustering_definition_core_v2_status_summary.csv"
AUDIT_SUMMARY_CSV = "nanoclustering_definition_core_v2_audit_summary.csv"
SUMMARY_JSON = "nanoclustering_definition_core_v2_summary.json"
REPORT_MD = "nanoclustering_definition_core_v2_report.md"
CONFIG_JSON = "nanoclustering_definition_core_v2_config.json"

ACCEPTED_V1_STATUS = "definition_core_v1_coherent"
RECOVERED_REFINEMENT_RESULT = "recovered_coherent_endpoint_vector_subfamily"
V2_COHERENT_STATUS = "definition_core_v2_coherent_primitive"
CLAIM_BOUNDARY = (
    "Definition-core v2 primitive registry only; no route execution, wall/pathway "
    "promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_definition_core_v2_registry"


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


def _value(row: pd.Series | dict[str, Any], column: str, default: Any = "") -> Any:
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return value


def _ordered(frame: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    remainder = [column for column in frame.columns if column not in preferred]
    return frame.loc[:, preferred + remainder]


def _v1_primitive_rows(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    accepted = registry[registry["definition_core_v1_status"].eq(ACCEPTED_V1_STATUS)].copy()
    for _, source in accepted.iterrows():
        rows.append(
            {
                "primitive_id": source["family_id"],
                "primitive_type": "v1_coherent_family",
                "primitive_origin": "accepted_definition_core_v1_family",
                "definition_core_v2_status": V2_COHERENT_STATUS,
                "definition_core_v2_read": (
                    "retained accepted support-local endpoint-vector family from v1"
                ),
                "source_family_id": source["family_id"],
                "source_definition_core_v1_status": source["definition_core_v1_status"],
                "source_family_vector_class": source["family_vector_class"],
                "primitive_vector_class": source["family_vector_class"],
                "primitive_coherence_status": source["coherence_status"],
                "branch": source["branch"],
                "ref_cluster_id": source["ref_cluster_id"],
                "boundary_family_tier": source["boundary_family_tier"],
                "definition_readiness": source["definition_readiness"],
                "event_count": int(source["event_count"]),
                "source_event_count": int(source["event_count"]),
                "event_count_share_of_source_family": 1.0,
                "source_ref_unit_count": int(source["ref_unit_count"]),
                "source_ref_weight_sum": int(source["ref_weight_sum"]),
                "decomposition_axis": "none_v1_coherent_family",
                "decomposition_axis_columns": "",
                "decomposition_key": "source_family_identity",
                "dominant_split_vector_class": source["dominant_split_vector_class"],
                "dominant_split_vector_class_share": float(
                    source["dominant_split_vector_class_share"]
                ),
                "dominant_host_context_class": source["dominant_host_context_class"],
                "dominant_host_context_class_share": float(
                    source["dominant_host_context_class_share"]
                ),
                "dominant_shape_core_signature": _value(
                    source, "dominant_shape_core_signature"
                ),
                "dominant_shape_core_signature_share": float(
                    source["dominant_shape_core_signature_share"]
                ),
                "dominant_host_handle_id": _value(source, "dominant_host_handle_id"),
                "dominant_host_handle_share": float(source["dominant_host_handle_share"]),
                "top1_segment_share_median": float(source["top1_segment_share_median"]),
                "top1_segment_share_iqr": float(source["top1_segment_share_iqr"]),
                "top2_segment_share_median": float(source["top2_segment_share_median"]),
                "top2_segment_share_iqr": float(source["top2_segment_share_iqr"]),
                "effective_segment_count_median": float(
                    source["effective_segment_count_median"]
                ),
                "effective_segment_count_iqr": float(source["effective_segment_count_iqr"]),
                "split_vector_class_counts": source["split_vector_class_counts"],
                "host_context_class_counts": source["host_context_class_counts"],
                "boundary_pattern_counts": source["boundary_pattern_counts"],
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _recovered_primitive_rows(
    *,
    registry: pd.DataFrame,
    subfamily_rows: pd.DataFrame,
) -> pd.DataFrame:
    source_lookup = {
        str(row["family_id"]): row for _, row in registry.set_index("family_id", drop=False).iterrows()
    }
    recovered = subfamily_rows[
        subfamily_rows["definition_refinement_result"].eq(RECOVERED_REFINEMENT_RESULT)
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, subfamily in recovered.iterrows():
        source = source_lookup[str(subfamily["source_family_id"])]
        rows.append(
            {
                "primitive_id": subfamily["subfamily_id"],
                "primitive_type": "recovered_coherent_subfamily",
                "primitive_origin": "recovered_from_definition_core_v1_refinement_queue",
                "definition_core_v2_status": V2_COHERENT_STATUS,
                "definition_core_v2_read": (
                    "promoted primary-axis recovered coherent endpoint-vector subfamily"
                ),
                "source_family_id": subfamily["source_family_id"],
                "source_definition_core_v1_status": subfamily[
                    "source_definition_core_v1_status"
                ],
                "source_family_vector_class": subfamily["source_family_vector_class"],
                "primitive_vector_class": subfamily["subfamily_vector_class"],
                "primitive_coherence_status": subfamily["subfamily_coherence_status"],
                "branch": subfamily["branch"],
                "ref_cluster_id": source["ref_cluster_id"],
                "boundary_family_tier": subfamily["boundary_family_tier"],
                "definition_readiness": source["definition_readiness"],
                "event_count": int(subfamily["event_count"]),
                "source_event_count": int(subfamily["source_event_count"]),
                "event_count_share_of_source_family": float(
                    subfamily["event_count_share_of_source_family"]
                ),
                "source_ref_unit_count": int(source["ref_unit_count"]),
                "source_ref_weight_sum": int(subfamily["source_ref_weight_sum"]),
                "decomposition_axis": subfamily["decomposition_axis"],
                "decomposition_axis_columns": subfamily["decomposition_axis_columns"],
                "decomposition_key": subfamily["decomposition_key"],
                "dominant_split_vector_class": subfamily["dominant_split_vector_class"],
                "dominant_split_vector_class_share": float(
                    subfamily["dominant_split_vector_class_share"]
                ),
                "dominant_host_context_class": subfamily["dominant_host_context_class"],
                "dominant_host_context_class_share": float(
                    subfamily["dominant_host_context_class_share"]
                ),
                "dominant_shape_core_signature": subfamily[
                    "dominant_shape_core_signature"
                ],
                "dominant_shape_core_signature_share": float(
                    subfamily["dominant_shape_core_signature_share"]
                ),
                "dominant_host_handle_id": subfamily["dominant_host_handle_id"],
                "dominant_host_handle_share": float(subfamily["dominant_host_handle_share"]),
                "top1_segment_share_median": float(subfamily["top1_segment_share_median"]),
                "top1_segment_share_iqr": float(subfamily["top1_segment_share_iqr"]),
                "top2_segment_share_median": float(subfamily["top2_segment_share_median"]),
                "top2_segment_share_iqr": float(subfamily["top2_segment_share_iqr"]),
                "effective_segment_count_median": float(
                    subfamily["effective_segment_count_median"]
                ),
                "effective_segment_count_iqr": float(
                    subfamily["effective_segment_count_iqr"]
                ),
                "split_vector_class_counts": subfamily["split_vector_class_counts"],
                "host_context_class_counts": subfamily["host_context_class_counts"],
                "boundary_pattern_counts": subfamily["boundary_pattern_counts"],
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _primitive_registry(
    *,
    registry: pd.DataFrame,
    subfamily_rows: pd.DataFrame,
) -> pd.DataFrame:
    primitive_registry = pd.concat(
        [
            _v1_primitive_rows(registry),
            _recovered_primitive_rows(registry=registry, subfamily_rows=subfamily_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    if primitive_registry["primitive_id"].duplicated().any():
        duplicates = primitive_registry[
            primitive_registry["primitive_id"].duplicated(keep=False)
        ]["primitive_id"].tolist()
        raise ValueError(f"duplicate primitive_id values: {duplicates[:10]}")
    preferred = [
        "primitive_id",
        "primitive_type",
        "primitive_origin",
        "definition_core_v2_status",
        "definition_core_v2_read",
        "source_family_id",
        "source_definition_core_v1_status",
        "source_family_vector_class",
        "primitive_vector_class",
        "primitive_coherence_status",
        "branch",
        "ref_cluster_id",
        "boundary_family_tier",
        "definition_readiness",
        "event_count",
        "source_event_count",
        "event_count_share_of_source_family",
        "source_ref_unit_count",
        "source_ref_weight_sum",
        "decomposition_axis",
        "decomposition_axis_columns",
        "decomposition_key",
        "dominant_split_vector_class",
        "dominant_split_vector_class_share",
        "dominant_host_context_class",
        "dominant_host_context_class_share",
        "dominant_shape_core_signature",
        "dominant_shape_core_signature_share",
        "dominant_host_handle_id",
        "dominant_host_handle_share",
        "top1_segment_share_median",
        "top1_segment_share_iqr",
        "top2_segment_share_median",
        "top2_segment_share_iqr",
        "effective_segment_count_median",
        "effective_segment_count_iqr",
        "split_vector_class_counts",
        "host_context_class_counts",
        "boundary_pattern_counts",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    return _ordered(
        primitive_registry.sort_values(
            [
                "primitive_type",
                "boundary_family_tier",
                "primitive_vector_class",
                "event_count",
                "primitive_id",
            ],
            ascending=[False, True, True, False, True],
        ),
        preferred,
    )


def _primitive_event_rows(
    *,
    registry: pd.DataFrame,
    primitive_registry: pd.DataFrame,
    coherence_events: pd.DataFrame,
    subfamily_event_rows: pd.DataFrame,
    subfamily_rows: pd.DataFrame,
) -> pd.DataFrame:
    accepted_family_ids = set(
        registry[registry["definition_core_v1_status"].eq(ACCEPTED_V1_STATUS)][
            "family_id"
        ].astype(str)
    )
    v1_events = coherence_events[
        coherence_events["family_id"].astype(str).isin(accepted_family_ids)
    ].copy()
    v1_meta = primitive_registry[
        primitive_registry["primitive_type"].eq("v1_coherent_family")
    ][
        [
            "primitive_id",
            "primitive_type",
            "definition_core_v2_status",
            "source_family_id",
            "source_definition_core_v1_status",
            "primitive_vector_class",
            "primitive_coherence_status",
        ]
    ]
    v1_events = v1_events.merge(
        v1_meta,
        left_on="family_id",
        right_on="source_family_id",
        how="left",
        validate="many_to_one",
    )

    recovered_events = subfamily_event_rows[
        subfamily_event_rows["definition_refinement_result"].eq(
            RECOVERED_REFINEMENT_RESULT
        )
    ].copy()
    subfamily_meta = subfamily_rows[
        [
            "subfamily_id",
            "source_definition_core_v1_status",
            "subfamily_vector_class",
            "subfamily_coherence_status",
        ]
    ].rename(
        columns={
            "subfamily_vector_class": "primitive_vector_class",
            "subfamily_coherence_status": "primitive_coherence_status",
        }
    )
    recovered_events = recovered_events.merge(
        subfamily_meta,
        on="subfamily_id",
        how="left",
        validate="many_to_one",
    )
    recovered_events["primitive_id"] = recovered_events["subfamily_id"]
    recovered_events["primitive_type"] = "recovered_coherent_subfamily"
    recovered_events["definition_core_v2_status"] = V2_COHERENT_STATUS

    event_rows = pd.concat([v1_events, recovered_events], ignore_index=True, sort=False)
    event_rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    event_rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    event_rows["quality_cost_status"] = QUALITY_COST_STATUS
    event_rows["claim_boundary"] = CLAIM_BOUNDARY
    event_rows["definition_core_v2_event_scope"] = "coherent_primitive_event"
    duplicate_mask = event_rows.duplicated(["source_family_id", "event_id"], keep=False)
    if duplicate_mask.any():
        duplicates = event_rows.loc[duplicate_mask, ["source_family_id", "event_id"]].head(10)
        raise ValueError(f"duplicate v2 event mappings:\n{duplicates}")
    expected_event_count = int(primitive_registry["event_count"].sum())
    if len(event_rows) != expected_event_count:
        raise ValueError(
            f"primitive event rows ({len(event_rows)}) != primitive event_count sum "
            f"({expected_event_count})"
        )
    preferred = [
        "primitive_id",
        "primitive_type",
        "definition_core_v2_status",
        "definition_core_v2_event_scope",
        "source_family_id",
        "source_definition_core_v1_status",
        "primitive_vector_class",
        "primitive_coherence_status",
        "event_id",
        "family_id",
        "branch",
        "boundary_family_tier",
        "split_vector_class",
        "host_context_class",
        "shape_core_signature",
        "shape_signature",
        "host_signature",
        "comparison_seed",
        "boundary_pattern",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    return _ordered(event_rows, preferred)


def _audit_queue_rows(
    *,
    subfamily_rows: pd.DataFrame,
    family_decomposition: pd.DataFrame,
    primitive_registry: pd.DataFrame,
) -> pd.DataFrame:
    recovered_source_ids = set(
        primitive_registry[
            primitive_registry["primitive_type"].eq("recovered_coherent_subfamily")
        ]["source_family_id"].astype(str)
    )
    residual_subfamilies = subfamily_rows[
        subfamily_rows["definition_refinement_result"].ne(RECOVERED_REFINEMENT_RESULT)
    ].copy()
    residual_subfamilies["audit_id"] = residual_subfamilies["subfamily_id"]
    residual_subfamilies["audit_row_type"] = "primary_subfamily_not_promoted"
    residual_subfamilies["definition_core_v2_audit_status"] = residual_subfamilies[
        "definition_refinement_result"
    ]
    residual_subfamilies["event_count_basis"] = "primary_subfamily_residual_additive"

    family_context = family_decomposition[
        ~family_decomposition["family_id"].astype(str).isin(recovered_source_ids)
    ].copy()
    family_context["audit_id"] = (
        family_context["family_id"] + "__source_family_without_v2_primary_recovery"
    )
    family_context["audit_row_type"] = "source_family_without_v2_primary_recovery"
    family_context["definition_core_v2_audit_status"] = family_context[
        "family_refinement_read"
    ]
    family_context["event_count"] = family_context["source_event_count"].astype(int)
    family_context["source_family_id"] = family_context["family_id"]
    family_context["source_definition_core_v1_status"] = family_context[
        "definition_core_v1_status"
    ]
    family_context["event_count_basis"] = "source_family_context_not_additive"

    common_cols = [
        "audit_id",
        "audit_row_type",
        "definition_core_v2_audit_status",
        "event_count_basis",
        "source_family_id",
        "source_definition_core_v1_status",
        "boundary_family_tier",
        "event_count",
        "decomposition_axis",
        "decomposition_axis_columns",
        "decomposition_key",
        "subfamily_coherence_status",
        "definition_refinement_result",
        "primary_axis",
        "best_axis",
        "primary_recovered_coherent_event_count",
        "best_axis_recovered_coherent_event_count",
        "family_refinement_read",
    ]
    for column in common_cols:
        if column not in residual_subfamilies:
            residual_subfamilies[column] = ""
        if column not in family_context:
            family_context[column] = ""
    audit_rows = pd.concat(
        [residual_subfamilies[common_cols], family_context[common_cols]],
        ignore_index=True,
        sort=False,
    )
    audit_rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    audit_rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    audit_rows["quality_cost_status"] = QUALITY_COST_STATUS
    audit_rows["claim_boundary"] = CLAIM_BOUNDARY
    return audit_rows.sort_values(
        [
            "audit_row_type",
            "source_definition_core_v1_status",
            "definition_core_v2_audit_status",
            "event_count",
            "audit_id",
        ],
        ascending=[True, True, True, False, True],
    )


def _status_summary(primitive_registry: pd.DataFrame) -> pd.DataFrame:
    rows = (
        primitive_registry.groupby(
            [
                "primitive_type",
                "boundary_family_tier",
                "primitive_vector_class",
                "definition_core_v2_status",
            ],
            as_index=False,
        )
        .agg(
            primitive_count=("primitive_id", "size"),
            source_family_count=("source_family_id", "nunique"),
            event_count_sum=("event_count", "sum"),
            median_event_count=("event_count", "median"),
            median_split_class_share=("dominant_split_vector_class_share", "median"),
            median_host_context_share=("dominant_host_context_class_share", "median"),
            median_shape_core_share=("dominant_shape_core_signature_share", "median"),
            median_host_handle_share=("dominant_host_handle_share", "median"),
        )
        .sort_values(
            ["primitive_type", "boundary_family_tier", "event_count_sum"],
            ascending=[False, True, False],
        )
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _audit_summary(audit_rows: pd.DataFrame) -> pd.DataFrame:
    rows = (
        audit_rows.groupby(
            [
                "audit_row_type",
                "event_count_basis",
                "source_definition_core_v1_status",
                "definition_core_v2_audit_status",
            ],
            as_index=False,
        )
        .agg(
            audit_row_count=("audit_id", "size"),
            source_family_count=("source_family_id", "nunique"),
            event_count_sum=("event_count", "sum"),
            median_event_count=("event_count", "median"),
        )
        .sort_values(
            ["audit_row_type", "source_definition_core_v1_status", "event_count_sum"],
            ascending=[True, True, False],
        )
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


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
    primitive_registry: pd.DataFrame,
    primitive_event_rows: pd.DataFrame,
    audit_rows: pd.DataFrame,
    status_summary: pd.DataFrame,
    audit_summary: pd.DataFrame,
    full_family_count: int,
    full_event_count: int,
) -> None:
    primitive_source_family_count = primitive_registry["source_family_id"].nunique()
    residual_subfamily_audit = audit_rows[
        audit_rows["event_count_basis"].eq("primary_subfamily_residual_additive")
    ]
    family_context_audit = audit_rows[
        audit_rows["event_count_basis"].eq("source_family_context_not_additive")
    ]
    event_coverage = len(primitive_event_rows) / float(full_event_count)
    family_coverage = primitive_source_family_count / float(full_family_count)
    text = [
        "# NanoClustering Definition-Core V2 Primitive Registry",
        "",
        f"- primitive_rows: `{len(primitive_registry)}`",
        f"- primitive_event_rows: `{len(primitive_event_rows)}`",
        f"- primitive_source_families: `{primitive_source_family_count}` of `{full_family_count}`",
        f"- event_coverage: `{len(primitive_event_rows)}` of `{full_event_count}` (`{event_coverage:.6f}`)",
        f"- source_family_coverage: `{primitive_source_family_count}` of `{full_family_count}` (`{family_coverage:.6f}`)",
        f"- residual_primary_subfamily_audit_events: `{int(residual_subfamily_audit['event_count'].sum())}`",
        f"- source_families_without_v2_primary_recovery: `{family_context_audit['source_family_id'].nunique()}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Primitive Status Summary",
        "",
        _markdown_table(
            status_summary,
            [
                "primitive_type",
                "boundary_family_tier",
                "primitive_vector_class",
                "primitive_count",
                "source_family_count",
                "event_count_sum",
                "median_event_count",
            ],
            max_rows=40,
        ),
        "",
        "## Audit Summary",
        "",
        _markdown_table(
            audit_summary,
            [
                "audit_row_type",
                "event_count_basis",
                "source_definition_core_v1_status",
                "definition_core_v2_audit_status",
                "audit_row_count",
                "source_family_count",
                "event_count_sum",
            ],
            max_rows=40,
        ),
        "",
        "## Source Families Without V2 Primary Recovery",
        "",
        _markdown_table(
            family_context_audit,
            [
                "source_family_id",
                "source_definition_core_v1_status",
                "definition_core_v2_audit_status",
                "event_count",
                "primary_axis",
                "best_axis",
                "primary_recovered_coherent_event_count",
                "best_axis_recovered_coherent_event_count",
            ],
            max_rows=30,
        ),
        "",
        "## Read",
        "",
        "- A v2 primitive is either an accepted v1 coherent family or a primary-axis recovered coherent endpoint-vector subfamily.",
        "- Alternative-axis recoveries are kept in audit until the primary decomposition rule is revised explicitly.",
        "- Singleton/tiny partitions remain diagnostic rows and do not inflate the primitive count.",
        "- This registry is still membership-derived endpoint cartography. It does not establish final global basins, wall/pathway traversal, basin quality, or route cost.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    v1_registry_dir: Path,
    refinement_dir: Path,
    coherence_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    registry = _read_csv(v1_registry_dir / V1_FAMILY_REGISTRY_CSV)
    subfamily_rows = _read_csv(refinement_dir / SUBFAMILY_ROWS_CSV)
    subfamily_event_rows = _read_csv(refinement_dir / SUBFAMILY_EVENT_ROWS_CSV)
    family_decomposition = _read_csv(refinement_dir / FAMILY_DECOMPOSITION_ROWS_CSV)
    coherence_events = _read_csv(coherence_dir / COHERENCE_EVENT_ROWS_CSV)

    primitive_registry = _primitive_registry(
        registry=registry,
        subfamily_rows=subfamily_rows,
    )
    primitive_event_rows = _primitive_event_rows(
        registry=registry,
        primitive_registry=primitive_registry,
        coherence_events=coherence_events,
        subfamily_event_rows=subfamily_event_rows,
        subfamily_rows=subfamily_rows,
    )
    audit_rows = _audit_queue_rows(
        subfamily_rows=subfamily_rows,
        family_decomposition=family_decomposition,
        primitive_registry=primitive_registry,
    )
    status_summary = _status_summary(primitive_registry)
    audit_summary = _audit_summary(audit_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(primitive_registry, output_dir / PRIMITIVE_REGISTRY_CSV)
    _write_csv(primitive_event_rows, output_dir / PRIMITIVE_EVENT_ROWS_CSV)
    _write_csv(audit_rows, output_dir / AUDIT_QUEUE_ROWS_CSV)
    _write_csv(status_summary, output_dir / STATUS_SUMMARY_CSV)
    _write_csv(audit_summary, output_dir / AUDIT_SUMMARY_CSV)
    _write_report(
        output_dir=output_dir,
        primitive_registry=primitive_registry,
        primitive_event_rows=primitive_event_rows,
        audit_rows=audit_rows,
        status_summary=status_summary,
        audit_summary=audit_summary,
        full_family_count=len(registry),
        full_event_count=len(coherence_events),
    )

    v1_primitive_rows = primitive_registry[
        primitive_registry["primitive_type"].eq("v1_coherent_family")
    ]
    recovered_primitive_rows = primitive_registry[
        primitive_registry["primitive_type"].eq("recovered_coherent_subfamily")
    ]
    residual_subfamily_audit = audit_rows[
        audit_rows["event_count_basis"].eq("primary_subfamily_residual_additive")
    ]
    family_context_audit = audit_rows[
        audit_rows["event_count_basis"].eq("source_family_context_not_additive")
    ]
    summary = {
        "ok": True,
        "v1_registry_dir": _rel(v1_registry_dir),
        "refinement_dir": _rel(refinement_dir),
        "coherence_dir": _rel(coherence_dir),
        "output_dir": _rel(output_dir),
        "full_definition_core_family_count": int(len(registry)),
        "full_definition_core_event_count": int(len(coherence_events)),
        "v1_coherent_family_count": int(len(v1_primitive_rows)),
        "v1_coherent_event_count": int(v1_primitive_rows["event_count"].sum()),
        "recovered_coherent_subfamily_count": int(len(recovered_primitive_rows)),
        "recovered_coherent_source_family_count": int(
            recovered_primitive_rows["source_family_id"].nunique()
        ),
        "recovered_coherent_event_count": int(recovered_primitive_rows["event_count"].sum()),
        "primitive_row_count": int(len(primitive_registry)),
        "primitive_event_row_count": int(len(primitive_event_rows)),
        "primitive_source_family_count": int(primitive_registry["source_family_id"].nunique()),
        "source_family_coverage_share": (
            primitive_registry["source_family_id"].nunique() / float(len(registry))
        ),
        "event_coverage_share": len(primitive_event_rows) / float(len(coherence_events)),
        "audit_primary_subfamily_row_count": int(len(residual_subfamily_audit)),
        "audit_primary_subfamily_event_count": int(residual_subfamily_audit["event_count"].sum()),
        "source_family_without_v2_primary_recovery_count": int(
            family_context_audit["source_family_id"].nunique()
        ),
        "primitive_type_counts": _count(primitive_registry, "primitive_type"),
        "primitive_status_counts": _count(primitive_registry, "definition_core_v2_status"),
        "audit_status_counts": _count(audit_rows, "definition_core_v2_audit_status"),
        "family_context_audit_status_counts": _count(
            family_context_audit,
            "definition_core_v2_audit_status",
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "primitive_registry_csv": _rel(output_dir / PRIMITIVE_REGISTRY_CSV),
            "primitive_event_rows_csv": _rel(output_dir / PRIMITIVE_EVENT_ROWS_CSV),
            "audit_queue_rows_csv": _rel(output_dir / AUDIT_QUEUE_ROWS_CSV),
            "status_summary_csv": _rel(output_dir / STATUS_SUMMARY_CSV),
            "audit_summary_csv": _rel(output_dir / AUDIT_SUMMARY_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "v1_registry_dir": _rel(v1_registry_dir),
        "refinement_dir": _rel(refinement_dir),
        "coherence_dir": _rel(coherence_dir),
        "output_dir": _rel(output_dir),
        "accepted_v1_status": ACCEPTED_V1_STATUS,
        "recovered_refinement_result": RECOVERED_REFINEMENT_RESULT,
        "definition_core_v2_status": V2_COHERENT_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "audit_promotion_rule": (
            "Promote accepted v1 coherent families and primary-axis recovered "
            "coherent endpoint-vector subfamilies only."
        ),
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
    parser.add_argument("--v1-registry-dir", type=Path, default=DEFAULT_V1_REGISTRY_DIR)
    parser.add_argument("--refinement-dir", type=Path, default=DEFAULT_REFINEMENT_DIR)
    parser.add_argument("--coherence-dir", type=Path, default=DEFAULT_COHERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        v1_registry_dir=args.v1_registry_dir.resolve(),
        refinement_dir=args.refinement_dir.resolve(),
        coherence_dir=args.coherence_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
