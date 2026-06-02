#!/usr/bin/env python3
"""Analyze NanoClustering definition-core v2.1 internals in detail.

This reads the v2.1 registry and membership-derived event-vector rows to inspect
confidence-tier concentration, thin-support distribution, axis-exception
best-axis subfamilies, and residual definition queues. It does not run
clustering, execute optimizer routes, promote wall/pathway claims, or inspect
basin quality/cost.
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
DEFAULT_V2_1_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_1_registry_20260531"
)
DEFAULT_COHERENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_full_basin_vector_coherence_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_1_detail_review_20260531"
)

V2_1_PRIMITIVE_REGISTRY_CSV = (
    "nanoclustering_definition_core_v2_1_primitive_registry.csv"
)
V2_1_AXIS_EXCEPTION_LEDGER_CSV = (
    "nanoclustering_definition_core_v2_1_axis_exception_ledger.csv"
)
V2_1_RESIDUAL_DEFINITION_QUEUE_CSV = (
    "nanoclustering_definition_core_v2_1_residual_definition_queue.csv"
)
COHERENCE_EVENT_ROWS_CSV = "nanoclustering_basin_vector_coherence_event_rows.csv"

CONFIDENCE_DETAIL_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_confidence_detail_rows.csv"
)
THIN_SUPPORT_SOURCE_FAMILY_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_thin_support_source_family_rows.csv"
)
AXIS_EXCEPTION_BEST_AXIS_SUBFAMILY_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_axis_exception_best_axis_subfamily_rows.csv"
)
AXIS_EXCEPTION_BEST_AXIS_EVENT_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_axis_exception_best_axis_event_rows.csv"
)
RESIDUAL_QUEUE_DETAIL_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_residual_queue_detail_rows.csv"
)
SUMMARY_JSON = "nanoclustering_definition_core_v2_1_detail_review_summary.json"
REPORT_MD = "nanoclustering_definition_core_v2_1_detail_review_report.md"
CONFIG_JSON = "nanoclustering_definition_core_v2_1_detail_review_config.json"

MIN_SUBFAMILY_EVENTS = 2
CLAIM_BOUNDARY = (
    "Definition-core v2.1 detail review only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_definition_core_v2_1_detail_review"

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


def _join_unique(values: pd.Series, *, limit: int = 20) -> str:
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


def _axis_key(row: pd.Series, axis: str) -> str:
    return "||".join(str(row[column]) for column in AXIS_COLUMNS[axis])


def _axis_key_label(axis: str) -> str:
    return "+".join(AXIS_COLUMNS[axis])


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


def _best_axis_definition_result(row: dict[str, Any]) -> str:
    if int(row["event_count"]) < MIN_SUBFAMILY_EVENTS:
        return "best_axis_tiny_subfamily_not_promoted"
    coherence_status = str(row["best_axis_subfamily_coherence_status"])
    if coherence_status == "coherent_vector_and_host_subfamily":
        return "best_axis_recovered_coherent_endpoint_vector_subfamily_not_promoted"
    if coherence_status == "coherent_class_with_numeric_variation_subfamily":
        return "best_axis_numeric_stress_subfamily_not_promoted"
    return "best_axis_still_requires_definition_refinement"


def _confidence_detail_rows(registry: pd.DataFrame) -> pd.DataFrame:
    rows = registry.copy()
    rows["downstream_definition_readiness"] = rows["definition_confidence_tier"].map(
        {
            "v2_1_family_coherent_confidence": "ready_as_family_coherent_definition_surface",
            "v2_1_deep_recovered_confidence": "ready_as_high_confidence_recovered_surface",
            "v2_1_moderate_recovered_confidence": "ready_as_moderate_confidence_recovered_surface",
            "v2_1_thin_recovered_confidence": "usable_but_thin_confidence_surface",
            "v2_1_axis_caveat_primitive": "usable_with_axis_caveat_surface",
        }
    ).fillna("definition_surface_review_required")
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _thin_support_source_family_rows(registry: pd.DataFrame) -> pd.DataFrame:
    thin = registry[registry["definition_confidence_tier"].eq("v2_1_thin_recovered_confidence")]
    if thin.empty:
        return pd.DataFrame()
    rows = (
        thin.groupby(
            [
                "source_family_id",
                "source_definition_core_v1_status",
                "branch",
                "boundary_family_tier",
            ],
            as_index=False,
        )
        .agg(
            thin_primitive_count=("primitive_id", "size"),
            thin_event_count_sum=("event_count", "sum"),
            thin_vector_class_counts=("primitive_vector_class", _count_string),
            thin_decomposition_axis_counts=("decomposition_axis", _count_string),
            thin_primitive_ids=("primitive_id", _join_unique),
        )
        .sort_values(
            [
                "thin_primitive_count",
                "thin_event_count_sum",
                "source_family_id",
            ],
            ascending=[False, False, True],
        )
    )
    rows["thin_support_read"] = rows["thin_primitive_count"].map(
        lambda count: (
            "multi_thin_primitives_same_source_family"
            if int(count) > 1
            else "single_thin_primitive_source_family"
        )
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _exception_subfamily_row(
    *,
    family_events: pd.DataFrame,
    exception_row: pd.Series,
    axis: str,
    key: str,
    rank: int,
) -> dict[str, Any]:
    group = family_events[family_events["best_axis_key"].eq(key)].copy()
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
    row: dict[str, Any] = {
        "family_id": str(exception_row["family_id"]),
        "exception_status": exception_row["definition_core_v2_1_exception_status"],
        "registry_effect": exception_row["definition_core_v2_1_registry_effect"],
        "axis_decision_read": exception_row["axis_decision_read"],
        "primary_axis": exception_row["primary_axis"],
        "best_axis": axis,
        "best_axis_columns": _axis_key_label(axis),
        "best_axis_key": key,
        "best_axis_subfamily_id": (
            f"{exception_row['family_id']}__best_{axis}__sub{rank:02d}"
        ),
        "branch": first["branch"],
        "boundary_family_tier": first["boundary_family_tier"],
        "definition_core_v1_status": exception_row["definition_core_v1_status"],
        "source_family_vector_class": exception_row["family_vector_class"],
        "best_axis_subfamily_vector_class": _subfamily_vector_class(group),
        "event_count": int(len(group)),
        "event_count_share_of_source_family": int(len(group))
        / float(exception_row["source_event_count"]),
        "source_event_count": int(exception_row["source_event_count"]),
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
        "effective_segment_count_median": float(effective.median()),
        "effective_segment_count_iqr": _iqr(effective),
        "boundary_pattern_counts": _count_string(group["boundary_pattern"]),
        "comparison_seeds": _join_unique(group["comparison_seed"]),
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "quality_cost_status": QUALITY_COST_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    row["best_axis_subfamily_coherence_status"] = _coherence_status(row)
    row["best_axis_definition_result"] = _best_axis_definition_result(row)
    return row


def _exception_best_axis_rows(
    *,
    exceptions: pd.DataFrame,
    full_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subfamily_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    for _, exception_row in exceptions.iterrows():
        axis = str(exception_row["best_axis"])
        family_id = str(exception_row["family_id"])
        family_events = full_events[full_events["family_id"].astype(str).eq(family_id)].copy()
        if family_events.empty:
            raise ValueError(f"missing coherence events for exception family {family_id}")
        family_events["best_axis"] = axis
        family_events["best_axis_columns"] = _axis_key_label(axis)
        family_events["best_axis_key"] = family_events.apply(
            lambda row: _axis_key(row, axis),
            axis=1,
        )
        size_order = (
            family_events.groupby("best_axis_key", as_index=False)
            .agg(event_count=("event_id", "size"))
            .sort_values(["event_count", "best_axis_key"], ascending=[False, True])
        )
        family_rows: list[dict[str, Any]] = []
        for rank, key in enumerate(size_order["best_axis_key"], start=1):
            family_rows.append(
                _exception_subfamily_row(
                    family_events=family_events,
                    exception_row=exception_row,
                    axis=axis,
                    key=key,
                    rank=rank,
                )
            )
        family_subfamilies = pd.DataFrame(family_rows)
        subfamily_rows.extend(family_rows)
        event_family = family_events.merge(
            family_subfamilies[
                [
                    "family_id",
                    "best_axis_key",
                    "best_axis_subfamily_id",
                    "best_axis_subfamily_coherence_status",
                    "best_axis_definition_result",
                ]
            ],
            on=["family_id", "best_axis_key"],
            how="left",
            validate="many_to_one",
        )
        event_family["exception_status"] = exception_row[
            "definition_core_v2_1_exception_status"
        ]
        event_family["registry_effect"] = exception_row[
            "definition_core_v2_1_registry_effect"
        ]
        event_family["axis_decision_read"] = exception_row["axis_decision_read"]
        event_family["claim_boundary"] = CLAIM_BOUNDARY
        event_frames.append(event_family)
    subfamilies = pd.DataFrame(subfamily_rows)
    events = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    return subfamilies, events


def _residual_queue_detail_rows(residual_queue: pd.DataFrame) -> pd.DataFrame:
    rows = residual_queue.copy()
    rows["residual_priority"] = rows["definition_core_v2_1_queue_status"].map(
        {
            "second_axis_definition_queue": "priority_1_second_axis_rule_design",
            "joint_axis_definition_queue": "priority_2_joint_axis_rule_design",
            "rule_edge_definition_queue": "priority_3_rule_edge_review",
            "support_depth_tiny_holdout": "holdout_support_depth_only",
        }
    ).fillna("definition_review")
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["residual_priority", "event_count", "source_family_id", "audit_id"],
        ascending=[True, False, True, True],
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
    confidence_detail: pd.DataFrame,
    thin_sources: pd.DataFrame,
    exception_subfamilies: pd.DataFrame,
    residual_detail: pd.DataFrame,
) -> None:
    confidence_rollup = (
        confidence_detail.groupby(
            [
                "definition_confidence_tier",
                "boundary_family_tier",
                "branch",
            ],
            as_index=False,
        )
        .agg(
            primitive_count=("primitive_id", "size"),
            event_count_sum=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
            median_event_count=("event_count", "median"),
        )
        .sort_values(["definition_confidence_tier", "event_count_sum"], ascending=[True, False])
    )
    vector_rollup = (
        confidence_detail.groupby(
            ["definition_confidence_tier", "primitive_vector_class"],
            as_index=False,
        )
        .agg(
            primitive_count=("primitive_id", "size"),
            event_count_sum=("event_count", "sum"),
        )
        .sort_values(["definition_confidence_tier", "event_count_sum"], ascending=[True, False])
    )
    exception_rollup = (
        exception_subfamilies.groupby(
            [
                "exception_status",
                "best_axis",
                "best_axis_definition_result",
            ],
            as_index=False,
        )
        .agg(
            subfamily_count=("best_axis_subfamily_id", "size"),
            event_count_sum=("event_count", "sum"),
            source_family_count=("family_id", "nunique"),
        )
        .sort_values(["exception_status", "event_count_sum"], ascending=[True, False])
    )
    residual_rollup = (
        residual_detail.groupby(
            [
                "residual_priority",
                "definition_core_v2_1_queue_status",
                "source_definition_core_v1_status",
            ],
            as_index=False,
        )
        .agg(
            queue_row_count=("audit_id", "size"),
            event_count_sum=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
        )
        .sort_values(["residual_priority", "event_count_sum"], ascending=[True, False])
    )
    text = [
        "# NanoClustering Definition-Core V2.1 Detail Review",
        "",
        f"- confidence_detail_rows: `{len(confidence_detail)}`",
        f"- thin_source_family_rows: `{len(thin_sources)}`",
        f"- axis_exception_best_axis_subfamily_rows: `{len(exception_subfamilies)}`",
        f"- residual_queue_detail_rows: `{len(residual_detail)}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Confidence By Tier, Boundary, And Branch",
        "",
        _markdown_table(
            confidence_rollup,
            [
                "definition_confidence_tier",
                "boundary_family_tier",
                "branch",
                "primitive_count",
                "event_count_sum",
                "source_family_count",
                "median_event_count",
            ],
            max_rows=40,
        ),
        "",
        "## Confidence By Vector Class",
        "",
        _markdown_table(
            vector_rollup,
            [
                "definition_confidence_tier",
                "primitive_vector_class",
                "primitive_count",
                "event_count_sum",
            ],
            max_rows=50,
        ),
        "",
        "## Thin Support Concentration",
        "",
        _markdown_table(
            thin_sources,
            [
                "source_family_id",
                "source_definition_core_v1_status",
                "branch",
                "boundary_family_tier",
                "thin_primitive_count",
                "thin_event_count_sum",
                "thin_vector_class_counts",
                "thin_support_read",
            ],
            max_rows=30,
        ),
        "",
        "## Axis Exception Best-Axis Subfamilies",
        "",
        _markdown_table(
            exception_rollup,
            [
                "exception_status",
                "best_axis",
                "best_axis_definition_result",
                "subfamily_count",
                "event_count_sum",
                "source_family_count",
            ],
            max_rows=40,
        ),
        "",
        "## Strong Exception Best-Axis Rows",
        "",
        _markdown_table(
            exception_subfamilies[
                exception_subfamilies["exception_status"].eq(
                    "strong_axis_exception_candidate_not_promoted"
                )
            ],
            [
                "family_id",
                "definition_core_v1_status",
                "best_axis",
                "best_axis_key",
                "event_count",
                "best_axis_subfamily_vector_class",
                "best_axis_subfamily_coherence_status",
                "best_axis_definition_result",
            ],
            max_rows=30,
        ),
        "",
        "## Residual Queue Detail",
        "",
        _markdown_table(
            residual_rollup,
            [
                "residual_priority",
                "definition_core_v2_1_queue_status",
                "source_definition_core_v1_status",
                "queue_row_count",
                "event_count_sum",
                "source_family_count",
            ],
            max_rows=30,
        ),
        "",
        "## Read",
        "",
        "- Thin recovered primitives are wide rather than dominated by one source family, so the main issue is support depth, not one pathological source.",
        "- The deep recovered tier is mostly a reliable subset of host-coherent split-mixed queues under the primary split-vector axis.",
        "- Strong axis exceptions are concrete enough for event-level exception-axis materialization, but remain outside v2.1 primitives.",
        "- The next definition work should focus on second-axis and joint-axis queues before any wall/pathway exploration.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    v2_1_dir: Path,
    coherence_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    registry = _read_csv(v2_1_dir / V2_1_PRIMITIVE_REGISTRY_CSV)
    exceptions = _read_csv(v2_1_dir / V2_1_AXIS_EXCEPTION_LEDGER_CSV)
    residual_queue = _read_csv(v2_1_dir / V2_1_RESIDUAL_DEFINITION_QUEUE_CSV)
    full_events = _read_csv(coherence_dir / COHERENCE_EVENT_ROWS_CSV)

    confidence_detail = _confidence_detail_rows(registry)
    thin_sources = _thin_support_source_family_rows(registry)
    exception_subfamilies, exception_events = _exception_best_axis_rows(
        exceptions=exceptions,
        full_events=full_events,
    )
    residual_detail = _residual_queue_detail_rows(residual_queue)

    recomputed = (
        exception_subfamilies[
            exception_subfamilies["best_axis_definition_result"].eq(
                "best_axis_recovered_coherent_endpoint_vector_subfamily_not_promoted"
            )
        ]
        .groupby("family_id", as_index=False)
        .agg(recomputed_best_recovered_event_count=("event_count", "sum"))
    )
    expected = exceptions[["family_id", "best_recovered_event_count"]].merge(
        recomputed,
        on="family_id",
        how="left",
        validate="one_to_one",
    )
    expected["recomputed_best_recovered_event_count"] = expected[
        "recomputed_best_recovered_event_count"
    ].fillna(0)
    mismatched = expected[
        expected["best_recovered_event_count"].astype(int).ne(
            expected["recomputed_best_recovered_event_count"].astype(int)
        )
    ]
    if not mismatched.empty:
        raise ValueError(f"best-axis recomputation mismatch:\n{mismatched}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(confidence_detail, output_dir / CONFIDENCE_DETAIL_ROWS_CSV)
    _write_csv(thin_sources, output_dir / THIN_SUPPORT_SOURCE_FAMILY_ROWS_CSV)
    _write_csv(exception_subfamilies, output_dir / AXIS_EXCEPTION_BEST_AXIS_SUBFAMILY_ROWS_CSV)
    _write_csv(exception_events, output_dir / AXIS_EXCEPTION_BEST_AXIS_EVENT_ROWS_CSV)
    _write_csv(residual_detail, output_dir / RESIDUAL_QUEUE_DETAIL_ROWS_CSV)
    _write_report(
        output_dir=output_dir,
        confidence_detail=confidence_detail,
        thin_sources=thin_sources,
        exception_subfamilies=exception_subfamilies,
        residual_detail=residual_detail,
    )

    strong_exception_rows = exception_subfamilies[
        exception_subfamilies["exception_status"].eq(
            "strong_axis_exception_candidate_not_promoted"
        )
    ]
    strong_recovered = strong_exception_rows[
        strong_exception_rows["best_axis_definition_result"].eq(
            "best_axis_recovered_coherent_endpoint_vector_subfamily_not_promoted"
        )
    ]
    summary = {
        "ok": True,
        "v2_1_dir": _rel(v2_1_dir),
        "coherence_dir": _rel(coherence_dir),
        "output_dir": _rel(output_dir),
        "confidence_detail_row_count": int(len(confidence_detail)),
        "thin_source_family_count": int(len(thin_sources)),
        "thin_multi_source_family_count": int(
            thin_sources["thin_primitive_count"].gt(1).sum()
        )
        if not thin_sources.empty
        else 0,
        "axis_exception_best_axis_subfamily_count": int(len(exception_subfamilies)),
        "axis_exception_best_axis_event_count": int(len(exception_events)),
        "strong_exception_best_axis_recovered_event_count": int(
            strong_recovered["event_count"].sum()
        ),
        "exception_best_axis_definition_result_counts": _count(
            exception_subfamilies,
            "best_axis_definition_result",
        ),
        "residual_queue_detail_row_count": int(len(residual_detail)),
        "residual_priority_counts": _count(residual_detail, "residual_priority"),
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "confidence_detail_rows_csv": _rel(output_dir / CONFIDENCE_DETAIL_ROWS_CSV),
            "thin_support_source_family_rows_csv": _rel(
                output_dir / THIN_SUPPORT_SOURCE_FAMILY_ROWS_CSV
            ),
            "axis_exception_best_axis_subfamily_rows_csv": _rel(
                output_dir / AXIS_EXCEPTION_BEST_AXIS_SUBFAMILY_ROWS_CSV
            ),
            "axis_exception_best_axis_event_rows_csv": _rel(
                output_dir / AXIS_EXCEPTION_BEST_AXIS_EVENT_ROWS_CSV
            ),
            "residual_queue_detail_rows_csv": _rel(output_dir / RESIDUAL_QUEUE_DETAIL_ROWS_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "v2_1_dir": _rel(v2_1_dir),
        "coherence_dir": _rel(coherence_dir),
        "output_dir": _rel(output_dir),
        "min_subfamily_events": MIN_SUBFAMILY_EVENTS,
        "axis_columns": AXIS_COLUMNS,
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
    parser.add_argument("--v2-1-dir", type=Path, default=DEFAULT_V2_1_DIR)
    parser.add_argument("--coherence-dir", type=Path, default=DEFAULT_COHERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        v2_1_dir=args.v2_1_dir.resolve(),
        coherence_dir=args.coherence_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
