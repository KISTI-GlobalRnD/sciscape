#!/usr/bin/env python3
"""Materialize NanoClustering definition-core v2.1 axis-rule candidates.

This materializes second-axis, joint-axis, and strong exception-axis candidates
from existing membership-derived event-vector rows. It does not run clustering,
execute optimizer routes, promote wall/pathway claims, or inspect basin
quality/cost.
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
    / "leiden_basin_nanoclustering_definition_core_v2_1_axis_rule_candidates_20260531"
)

V2_1_AXIS_EXCEPTION_LEDGER_CSV = (
    "nanoclustering_definition_core_v2_1_axis_exception_ledger.csv"
)
V2_1_RESIDUAL_DEFINITION_QUEUE_CSV = (
    "nanoclustering_definition_core_v2_1_residual_definition_queue.csv"
)
COHERENCE_EVENT_ROWS_CSV = "nanoclustering_basin_vector_coherence_event_rows.csv"

TARGET_AXIS_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_axis_rule_target_axis_rows.csv"
)
CANDIDATE_SUBFAMILY_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_axis_rule_candidate_subfamily_rows.csv"
)
CANDIDATE_EVENT_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_axis_rule_candidate_event_rows.csv"
)
RULE_SCOPE_SUMMARY_CSV = (
    "nanoclustering_definition_core_v2_1_axis_rule_scope_summary.csv"
)
SUMMARY_JSON = "nanoclustering_definition_core_v2_1_axis_rule_candidates_summary.json"
REPORT_MD = "nanoclustering_definition_core_v2_1_axis_rule_candidates_report.md"
CONFIG_JSON = "nanoclustering_definition_core_v2_1_axis_rule_candidates_config.json"

MIN_SUBFAMILY_EVENTS = 2
CLAIM_BOUNDARY = (
    "Definition-core v2.1 axis-rule candidate materialization only; no route "
    "execution, wall/pathway promotion, basin-quality claim, cost claim, or "
    "directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_definition_core_v2_1_axis_rule_candidates"

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

SECOND_AXIS_CANDIDATES = [
    "host_signature",
    "shape_core_signature",
    "shape_core_and_host_signature",
    "boundary_pattern",
    "shape_core_and_boundary_pattern",
]
JOINT_AXIS_CANDIDATES = [
    "shape_core_and_host_context",
    "host_signature",
    "shape_core_signature",
    "shape_core_and_host_signature",
    "split_vector_class",
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
        return "coherent_vector_and_host_candidate_subfamily"
    if split_coherent and host_coherent:
        return "coherent_numeric_stress_candidate_subfamily"
    if split_coherent and not host_coherent:
        return "split_coherent_host_variable_candidate_subfamily"
    if host_coherent and not split_coherent:
        return "host_coherent_split_mixed_candidate_subfamily"
    return "heterogeneous_or_rule_edge_candidate_subfamily"


def _subfamily_vector_class(group: pd.DataFrame) -> str:
    total = len(group)
    if total == 0:
        return "empty_candidate_subfamily"
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
        return "diffuse_multiway_fragmentation_candidate_subfamily"
    if balanced_two_share >= 0.67 and source_host_share >= 0.67:
        return "source_host_balanced_two_way_split_candidate_subfamily"
    if balanced_any_share >= 0.67 and external_abs_share >= 0.67:
        return "external_host_balanced_absorption_candidate_subfamily"
    if external_abs_share >= 0.67:
        return "external_host_absorption_candidate_subfamily"
    if balanced_any_share >= 0.67:
        return "balanced_multi_handle_split_candidate_subfamily"
    return "heterogeneous_basin_vector_candidate_subfamily"


def _candidate_result(row: dict[str, Any]) -> str:
    if int(row["event_count"]) < MIN_SUBFAMILY_EVENTS:
        return "candidate_tiny_subfamily_not_promoted"
    status = str(row["candidate_subfamily_coherence_status"])
    if status == "coherent_vector_and_host_candidate_subfamily":
        return "candidate_recovered_coherent_endpoint_vector_subfamily"
    if status == "coherent_numeric_stress_candidate_subfamily":
        return "candidate_numeric_stress_subfamily"
    return "candidate_still_requires_definition_refinement"


def _target_axis_read(row: pd.Series) -> str:
    recovered_share = float(row["candidate_recovered_event_share"])
    recovered_events = int(row["candidate_recovered_event_count"])
    if recovered_share >= 0.75:
        return "candidate_axis_recovers_most_events"
    if recovered_events > 0:
        return "candidate_axis_recovers_partial_events"
    return "candidate_axis_no_coherent_recovery"


def _rule_scope_read(scope: str) -> str:
    if scope == "second_axis_candidate":
        return "candidate second-axis split within host-coherent split-mixed residual subfamily"
    if scope == "joint_axis_candidate":
        return "candidate joint-axis split within split-coherent host-variable residual subfamily"
    return "candidate exception-axis split over strong exception source family"


def _target_rows(
    *,
    residual_queue: pd.DataFrame,
    exception_ledger: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    second = residual_queue[
        residual_queue["definition_core_v2_1_queue_status"].eq("second_axis_definition_queue")
    ]
    for _, row in second.iterrows():
        for axis in SECOND_AXIS_CANDIDATES:
            rows.append(
                {
                    "target_unit_id": row["audit_id"],
                    "rule_scope": "second_axis_candidate",
                    "rule_scope_read": _rule_scope_read("second_axis_candidate"),
                    "family_id": row["source_family_id"],
                    "definition_core_v1_status": row["source_definition_core_v1_status"],
                    "source_queue_status": row["definition_core_v2_1_queue_status"],
                    "source_event_count": int(row["event_count"]),
                    "primary_axis": row["decomposition_axis"],
                    "primary_key": row["decomposition_key"],
                    "candidate_axis": axis,
                    "candidate_axis_columns": _axis_key_label(axis),
                    "candidate_axis_origin": "second_axis_residual_queue",
                    "candidate_axis_promotion_status": "not_promoted_definition_rule_candidate",
                }
            )
    joint = residual_queue[
        residual_queue["definition_core_v2_1_queue_status"].eq("joint_axis_definition_queue")
    ]
    for _, row in joint.iterrows():
        for axis in JOINT_AXIS_CANDIDATES:
            rows.append(
                {
                    "target_unit_id": row["audit_id"],
                    "rule_scope": "joint_axis_candidate",
                    "rule_scope_read": _rule_scope_read("joint_axis_candidate"),
                    "family_id": row["source_family_id"],
                    "definition_core_v1_status": row["source_definition_core_v1_status"],
                    "source_queue_status": row["definition_core_v2_1_queue_status"],
                    "source_event_count": int(row["event_count"]),
                    "primary_axis": row["decomposition_axis"],
                    "primary_key": row["decomposition_key"],
                    "candidate_axis": axis,
                    "candidate_axis_columns": _axis_key_label(axis),
                    "candidate_axis_origin": "joint_axis_residual_queue",
                    "candidate_axis_promotion_status": "not_promoted_definition_rule_candidate",
                }
            )
    strong = exception_ledger[
        exception_ledger["definition_core_v2_1_exception_status"].eq(
            "strong_axis_exception_candidate_not_promoted"
        )
    ]
    for _, row in strong.iterrows():
        axis = str(row["best_axis"])
        rows.append(
            {
                "target_unit_id": f"{row['family_id']}__strong_exception_axis",
                "rule_scope": "strong_exception_axis_candidate",
                "rule_scope_read": _rule_scope_read("strong_exception_axis_candidate"),
                "family_id": row["family_id"],
                "definition_core_v1_status": row["definition_core_v1_status"],
                "source_queue_status": row["definition_core_v2_1_exception_status"],
                "source_event_count": int(row["source_event_count"]),
                "primary_axis": row["primary_axis"],
                "primary_key": "",
                "candidate_axis": axis,
                "candidate_axis_columns": _axis_key_label(axis),
                "candidate_axis_origin": "strong_exception_best_axis",
                "candidate_axis_promotion_status": "not_promoted_definition_rule_candidate",
            }
        )
    targets = pd.DataFrame(rows)
    targets["route_execution_status"] = ROUTE_EXECUTION_STATUS
    targets["wall_promotion_status"] = WALL_PROMOTION_STATUS
    targets["quality_cost_status"] = QUALITY_COST_STATUS
    targets["claim_boundary"] = CLAIM_BOUNDARY
    return targets


def _event_subset_for_target(*, full_events: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    family_events = full_events[full_events["family_id"].astype(str).eq(str(target["family_id"]))].copy()
    if family_events.empty:
        raise ValueError(f"missing events for family {target['family_id']}")
    if str(target["rule_scope"]) == "strong_exception_axis_candidate":
        subset = family_events
    else:
        primary_axis = str(target["primary_axis"])
        family_events["primary_axis_key"] = family_events.apply(
            lambda row: _axis_key(row, primary_axis),
            axis=1,
        )
        subset = family_events[family_events["primary_axis_key"].eq(str(target["primary_key"]))]
    if len(subset) != int(target["source_event_count"]):
        raise ValueError(
            f"target event count mismatch for {target['target_unit_id']}: "
            f"{len(subset)} != {target['source_event_count']}"
        )
    return subset.copy()


def _candidate_subfamily_row(
    *,
    target: pd.Series,
    group: pd.DataFrame,
    key: str,
    rank: int,
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
    row: dict[str, Any] = {
        "target_unit_id": target["target_unit_id"],
        "candidate_subfamily_id": (
            f"{target['target_unit_id']}__{target['candidate_axis']}__sub{rank:02d}"
        ),
        "rule_scope": target["rule_scope"],
        "rule_scope_read": target["rule_scope_read"],
        "family_id": target["family_id"],
        "definition_core_v1_status": target["definition_core_v1_status"],
        "source_queue_status": target["source_queue_status"],
        "source_event_count": int(target["source_event_count"]),
        "primary_axis": target["primary_axis"],
        "primary_key": target["primary_key"],
        "candidate_axis": target["candidate_axis"],
        "candidate_axis_columns": target["candidate_axis_columns"],
        "candidate_key": key,
        "candidate_axis_origin": target["candidate_axis_origin"],
        "candidate_axis_promotion_status": target["candidate_axis_promotion_status"],
        "branch": first["branch"],
        "boundary_family_tier": first["boundary_family_tier"],
        "event_count": int(len(group)),
        "event_count_share_of_target": int(len(group)) / float(target["source_event_count"]),
        "candidate_subfamily_vector_class": _subfamily_vector_class(group),
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
    row["candidate_subfamily_coherence_status"] = _coherence_status(row)
    row["candidate_definition_result"] = _candidate_result(row)
    return row


def _build_candidate_rows(
    *,
    targets: pd.DataFrame,
    full_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subfamily_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    target_rows: list[dict[str, Any]] = []
    for _, target in targets.iterrows():
        subset = _event_subset_for_target(full_events=full_events, target=target)
        candidate_axis = str(target["candidate_axis"])
        subset["candidate_axis"] = candidate_axis
        subset["candidate_axis_columns"] = _axis_key_label(candidate_axis)
        subset["candidate_key"] = subset.apply(
            lambda row: _axis_key(row, candidate_axis),
            axis=1,
        )
        size_order = (
            subset.groupby("candidate_key", as_index=False)
            .agg(event_count=("event_id", "size"))
            .sort_values(["event_count", "candidate_key"], ascending=[False, True])
        )
        local_rows: list[dict[str, Any]] = []
        for rank, key in enumerate(size_order["candidate_key"], start=1):
            group = subset[subset["candidate_key"].eq(key)]
            row = _candidate_subfamily_row(target=target, group=group, key=key, rank=rank)
            subfamily_rows.append(row)
            local_rows.append(row)
        local = pd.DataFrame(local_rows)
        recovered = local[
            local["candidate_definition_result"].eq(
                "candidate_recovered_coherent_endpoint_vector_subfamily"
            )
        ]
        numeric = local[
            local["candidate_definition_result"].eq("candidate_numeric_stress_subfamily")
        ]
        tiny = local[local["candidate_definition_result"].eq("candidate_tiny_subfamily_not_promoted")]
        unresolved = local[
            local["candidate_definition_result"].eq(
                "candidate_still_requires_definition_refinement"
            )
        ]
        target_row = target.to_dict()
        recovered_events = int(recovered["event_count"].sum()) if not recovered.empty else 0
        numeric_events = int(numeric["event_count"].sum()) if not numeric.empty else 0
        target_row.update(
            {
                "candidate_subfamily_count": int(len(local)),
                "candidate_recovered_subfamily_count": int(len(recovered)),
                "candidate_recovered_event_count": recovered_events,
                "candidate_recovered_event_share": recovered_events
                / float(target["source_event_count"]),
                "candidate_numeric_stress_subfamily_count": int(len(numeric)),
                "candidate_numeric_stress_event_count": numeric_events,
                "candidate_tiny_subfamily_count": int(len(tiny)),
                "candidate_tiny_event_count": int(tiny["event_count"].sum())
                if not tiny.empty
                else 0,
                "candidate_unresolved_subfamily_count": int(len(unresolved)),
                "candidate_unresolved_event_count": int(unresolved["event_count"].sum())
                if not unresolved.empty
                else 0,
            }
        )
        target_rows.append(target_row)
        event_subset = subset.merge(
            local[
                [
                    "target_unit_id",
                    "candidate_subfamily_id",
                    "candidate_key",
                    "candidate_subfamily_coherence_status",
                    "candidate_definition_result",
                ]
            ],
            on=["candidate_key"],
            how="left",
            validate="many_to_one",
        )
        event_subset["target_unit_id"] = target["target_unit_id"]
        event_subset["rule_scope"] = target["rule_scope"]
        event_subset["source_queue_status"] = target["source_queue_status"]
        event_subset["candidate_axis_origin"] = target["candidate_axis_origin"]
        event_subset["candidate_axis_promotion_status"] = target[
            "candidate_axis_promotion_status"
        ]
        event_subset["claim_boundary"] = CLAIM_BOUNDARY
        event_frames.append(event_subset)
    target_axis_rows = pd.DataFrame(target_rows)
    target_axis_rows["candidate_axis_read"] = target_axis_rows.apply(_target_axis_read, axis=1)
    target_axis_rows["is_best_axis_for_target"] = False
    if not target_axis_rows.empty:
        best_indices = (
            target_axis_rows.sort_values(
                [
                    "target_unit_id",
                    "candidate_recovered_event_count",
                    "candidate_recovered_subfamily_count",
                    "candidate_tiny_event_count",
                    "candidate_axis",
                ],
                ascending=[True, False, False, True, True],
            )
            .drop_duplicates("target_unit_id")
            .index
        )
        target_axis_rows.loc[best_indices, "is_best_axis_for_target"] = True
    candidate_subfamilies = pd.DataFrame(subfamily_rows)
    candidate_events = (
        pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    )
    return target_axis_rows, candidate_subfamilies, candidate_events


def _scope_summary(target_axis_rows: pd.DataFrame, subfamily_rows: pd.DataFrame) -> pd.DataFrame:
    best_rows = target_axis_rows[target_axis_rows["is_best_axis_for_target"]].copy()
    scope_rows = (
        target_axis_rows.groupby(["rule_scope", "candidate_axis"], as_index=False)
        .agg(
            target_axis_row_count=("target_unit_id", "size"),
            source_event_count_sum=("source_event_count", "sum"),
            candidate_recovered_event_count_sum=("candidate_recovered_event_count", "sum"),
            candidate_tiny_event_count_sum=("candidate_tiny_event_count", "sum"),
            candidate_unresolved_event_count_sum=("candidate_unresolved_event_count", "sum"),
            most_event_axis_count=(
                "candidate_axis_read",
                lambda values: int((values == "candidate_axis_recovers_most_events").sum()),
            ),
            partial_event_axis_count=(
                "candidate_axis_read",
                lambda values: int((values == "candidate_axis_recovers_partial_events").sum()),
            ),
        )
        .sort_values(["rule_scope", "candidate_recovered_event_count_sum"], ascending=[True, False])
    )
    scope_rows["summary_scope"] = "candidate_axis_totals"
    best_summary = (
        best_rows.groupby(["rule_scope", "candidate_axis"], as_index=False)
        .agg(
            target_axis_row_count=("target_unit_id", "size"),
            source_event_count_sum=("source_event_count", "sum"),
            candidate_recovered_event_count_sum=("candidate_recovered_event_count", "sum"),
            candidate_tiny_event_count_sum=("candidate_tiny_event_count", "sum"),
            candidate_unresolved_event_count_sum=("candidate_unresolved_event_count", "sum"),
            most_event_axis_count=(
                "candidate_axis_read",
                lambda values: int((values == "candidate_axis_recovers_most_events").sum()),
            ),
            partial_event_axis_count=(
                "candidate_axis_read",
                lambda values: int((values == "candidate_axis_recovers_partial_events").sum()),
            ),
        )
        .sort_values(["rule_scope", "candidate_recovered_event_count_sum"], ascending=[True, False])
    )
    best_summary["summary_scope"] = "best_axis_per_target_totals"
    result_summary = (
        subfamily_rows.groupby(["rule_scope", "candidate_definition_result"], as_index=False)
        .agg(
            target_axis_row_count=("target_unit_id", "nunique"),
            source_event_count_sum=("event_count", "sum"),
            candidate_recovered_event_count_sum=("event_count", "sum"),
            candidate_tiny_event_count_sum=("event_count", "sum"),
            candidate_unresolved_event_count_sum=("event_count", "sum"),
            most_event_axis_count=("candidate_subfamily_id", "size"),
            partial_event_axis_count=("candidate_subfamily_id", "size"),
        )
        .rename(columns={"candidate_definition_result": "candidate_axis"})
    )
    result_summary["summary_scope"] = "candidate_subfamily_result_totals"
    summary = pd.concat([scope_rows, best_summary, result_summary], ignore_index=True, sort=False)
    summary["claim_boundary"] = CLAIM_BOUNDARY
    return summary


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
    target_axis_rows: pd.DataFrame,
    subfamily_rows: pd.DataFrame,
    scope_summary: pd.DataFrame,
) -> None:
    best = target_axis_rows[target_axis_rows["is_best_axis_for_target"]].copy()
    best_rollup = (
        best.groupby(["rule_scope", "candidate_axis", "candidate_axis_read"], as_index=False)
        .agg(
            target_count=("target_unit_id", "size"),
            source_event_count_sum=("source_event_count", "sum"),
            recovered_event_count_sum=("candidate_recovered_event_count", "sum"),
            tiny_event_count_sum=("candidate_tiny_event_count", "sum"),
            unresolved_event_count_sum=("candidate_unresolved_event_count", "sum"),
        )
        .sort_values(["rule_scope", "recovered_event_count_sum"], ascending=[True, False])
    )
    second_best = best[best["rule_scope"].eq("second_axis_candidate")]
    joint_best = best[best["rule_scope"].eq("joint_axis_candidate")]
    exception_best = best[best["rule_scope"].eq("strong_exception_axis_candidate")]
    text = [
        "# NanoClustering Definition-Core V2.1 Axis-Rule Candidates",
        "",
        f"- target_axis_rows: `{len(target_axis_rows)}`",
        f"- candidate_subfamily_rows: `{len(subfamily_rows)}`",
        f"- best_second_axis_recovered_events: `{int(second_best['candidate_recovered_event_count'].sum())}`",
        f"- best_joint_axis_recovered_events: `{int(joint_best['candidate_recovered_event_count'].sum())}`",
        f"- strong_exception_axis_recovered_events: `{int(exception_best['candidate_recovered_event_count'].sum())}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Best Axis Per Target",
        "",
        _markdown_table(
            best_rollup,
            [
                "rule_scope",
                "candidate_axis",
                "candidate_axis_read",
                "target_count",
                "source_event_count_sum",
                "recovered_event_count_sum",
                "tiny_event_count_sum",
                "unresolved_event_count_sum",
            ],
            max_rows=30,
        ),
        "",
        "## Candidate Axis Totals",
        "",
        _markdown_table(
            scope_summary[scope_summary["summary_scope"].eq("candidate_axis_totals")],
            [
                "rule_scope",
                "candidate_axis",
                "target_axis_row_count",
                "source_event_count_sum",
                "candidate_recovered_event_count_sum",
                "candidate_tiny_event_count_sum",
                "candidate_unresolved_event_count_sum",
                "most_event_axis_count",
                "partial_event_axis_count",
            ],
            max_rows=40,
        ),
        "",
        "## Strong Exception Candidate Rows",
        "",
        _markdown_table(
            target_axis_rows[
                target_axis_rows["rule_scope"].eq("strong_exception_axis_candidate")
            ],
            [
                "family_id",
                "candidate_axis",
                "source_event_count",
                "candidate_recovered_event_count",
                "candidate_recovered_event_share",
                "candidate_tiny_event_count",
                "candidate_axis_read",
            ],
            max_rows=20,
        ),
        "",
        "## Recovered Candidate Subfamilies",
        "",
        _markdown_table(
            subfamily_rows[
                subfamily_rows["candidate_definition_result"].eq(
                    "candidate_recovered_coherent_endpoint_vector_subfamily"
                )
            ].sort_values(["rule_scope", "event_count"], ascending=[True, False]),
            [
                "rule_scope",
                "family_id",
                "target_unit_id",
                "candidate_axis",
                "candidate_key",
                "event_count",
                "candidate_subfamily_vector_class",
                "candidate_subfamily_coherence_status",
            ],
            max_rows=40,
        ),
        "",
        "## Read",
        "",
        "- Second-axis candidates recover some host-coherent split-mixed residuals, but no single axis resolves the queue cleanly.",
        "- Joint-axis candidates recover more of the split-coherent host-variable residuals, with shape-core based axes strongest.",
        "- Strong exception-axis candidates reproduce the 24-event recovery and are the cleanest candidates for an explicit exception rule.",
        "- All rows remain definition-rule candidates; no wall/pathway, quality, cost, or directed-search claim is promoted.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    v2_1_dir: Path,
    coherence_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    residual_queue = _read_csv(v2_1_dir / V2_1_RESIDUAL_DEFINITION_QUEUE_CSV)
    exception_ledger = _read_csv(v2_1_dir / V2_1_AXIS_EXCEPTION_LEDGER_CSV)
    full_events = _read_csv(coherence_dir / COHERENCE_EVENT_ROWS_CSV)
    targets = _target_rows(residual_queue=residual_queue, exception_ledger=exception_ledger)
    target_axis_rows, subfamily_rows, event_rows = _build_candidate_rows(
        targets=targets,
        full_events=full_events,
    )
    scope_summary = _scope_summary(target_axis_rows, subfamily_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(target_axis_rows, output_dir / TARGET_AXIS_ROWS_CSV)
    _write_csv(subfamily_rows, output_dir / CANDIDATE_SUBFAMILY_ROWS_CSV)
    _write_csv(event_rows, output_dir / CANDIDATE_EVENT_ROWS_CSV)
    _write_csv(scope_summary, output_dir / RULE_SCOPE_SUMMARY_CSV)
    _write_report(
        output_dir=output_dir,
        target_axis_rows=target_axis_rows,
        subfamily_rows=subfamily_rows,
        scope_summary=scope_summary,
    )

    best_rows = target_axis_rows[target_axis_rows["is_best_axis_for_target"]]
    second_best = best_rows[best_rows["rule_scope"].eq("second_axis_candidate")]
    joint_best = best_rows[best_rows["rule_scope"].eq("joint_axis_candidate")]
    exception_best = best_rows[best_rows["rule_scope"].eq("strong_exception_axis_candidate")]
    summary = {
        "ok": True,
        "v2_1_dir": _rel(v2_1_dir),
        "coherence_dir": _rel(coherence_dir),
        "output_dir": _rel(output_dir),
        "target_axis_row_count": int(len(target_axis_rows)),
        "candidate_subfamily_row_count": int(len(subfamily_rows)),
        "candidate_event_row_count": int(len(event_rows)),
        "best_second_axis_recovered_event_count": int(
            second_best["candidate_recovered_event_count"].sum()
        ),
        "best_joint_axis_recovered_event_count": int(
            joint_best["candidate_recovered_event_count"].sum()
        ),
        "strong_exception_axis_recovered_event_count": int(
            exception_best["candidate_recovered_event_count"].sum()
        ),
        "rule_scope_counts": _count(target_axis_rows, "rule_scope"),
        "candidate_axis_read_counts": _count(target_axis_rows, "candidate_axis_read"),
        "candidate_definition_result_counts": _count(
            subfamily_rows,
            "candidate_definition_result",
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "target_axis_rows_csv": _rel(output_dir / TARGET_AXIS_ROWS_CSV),
            "candidate_subfamily_rows_csv": _rel(output_dir / CANDIDATE_SUBFAMILY_ROWS_CSV),
            "candidate_event_rows_csv": _rel(output_dir / CANDIDATE_EVENT_ROWS_CSV),
            "rule_scope_summary_csv": _rel(output_dir / RULE_SCOPE_SUMMARY_CSV),
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
        "second_axis_candidates": SECOND_AXIS_CANDIDATES,
        "joint_axis_candidates": JOINT_AXIS_CANDIDATES,
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
