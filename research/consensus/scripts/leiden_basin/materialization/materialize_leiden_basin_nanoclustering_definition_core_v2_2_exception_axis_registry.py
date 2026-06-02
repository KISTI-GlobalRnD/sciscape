#!/usr/bin/env python3
"""Materialize the NanoClustering definition-core v2.2 exception-axis registry.

V2.2 is intentionally narrow: it inherits the v2.1 primitive registry and adds
only recovered coherent subfamilies from the strong exception-axis rule. It
does not run clustering, execute optimizer routes, promote wall/pathway claims,
or inspect basin quality/cost.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_V2_1_REGISTRY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_1_registry_20260531"
)
DEFAULT_AXIS_RULE_CANDIDATES_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_1_axis_rule_candidates_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_2_exception_axis_registry_20260531"
)

V2_1_PRIMITIVE_REGISTRY_CSV = (
    "nanoclustering_definition_core_v2_1_primitive_registry.csv"
)
V2_1_PRIMITIVE_EVENT_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_primitive_event_rows.csv"
)
V2_1_AXIS_EXCEPTION_LEDGER_CSV = (
    "nanoclustering_definition_core_v2_1_axis_exception_ledger.csv"
)
V2_1_RESIDUAL_DEFINITION_QUEUE_CSV = (
    "nanoclustering_definition_core_v2_1_residual_definition_queue.csv"
)
AXIS_RULE_TARGET_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_axis_rule_target_axis_rows.csv"
)
AXIS_RULE_SUBFAMILY_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_axis_rule_candidate_subfamily_rows.csv"
)
AXIS_RULE_EVENT_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_axis_rule_candidate_event_rows.csv"
)

V2_2_PRIMITIVE_REGISTRY_CSV = (
    "nanoclustering_definition_core_v2_2_primitive_registry.csv"
)
V2_2_PRIMITIVE_EVENT_ROWS_CSV = (
    "nanoclustering_definition_core_v2_2_primitive_event_rows.csv"
)
V2_2_EXCEPTION_AXIS_PRIMITIVE_ROWS_CSV = (
    "nanoclustering_definition_core_v2_2_exception_axis_primitive_rows.csv"
)
V2_2_EXCEPTION_AXIS_TINY_HOLDOUT_ROWS_CSV = (
    "nanoclustering_definition_core_v2_2_exception_axis_tiny_holdout_rows.csv"
)
V2_2_AXIS_EXCEPTION_LEDGER_CSV = (
    "nanoclustering_definition_core_v2_2_axis_exception_ledger.csv"
)
V2_2_RESIDUAL_DEFINITION_QUEUE_CSV = (
    "nanoclustering_definition_core_v2_2_residual_definition_queue.csv"
)
V2_2_CONFIDENCE_SUMMARY_CSV = (
    "nanoclustering_definition_core_v2_2_confidence_summary.csv"
)
V2_2_AXIS_RULE_SUMMARY_CSV = (
    "nanoclustering_definition_core_v2_2_axis_rule_summary.csv"
)
SUMMARY_JSON = "nanoclustering_definition_core_v2_2_exception_axis_summary.json"
REPORT_MD = "nanoclustering_definition_core_v2_2_exception_axis_report.md"
CONFIG_JSON = "nanoclustering_definition_core_v2_2_exception_axis_config.json"

RECOVERED_RESULT = "candidate_recovered_coherent_endpoint_vector_subfamily"
TINY_RESULT = "candidate_tiny_subfamily_not_promoted"
STRONG_RULE_SCOPE = "strong_exception_axis_candidate"
CLAIM_BOUNDARY = (
    "Definition-core v2.2 exception-axis registry only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search "
    "claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_definition_core_v2_2_exception_axis_registry"


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


def _ref_cluster_id(family_id: str) -> int | None:
    match = re.search(r"_ref(\d+)$", str(family_id))
    return int(match.group(1)) if match else None


def _exception_support_depth_tier(event_count: int) -> str:
    if event_count >= 5:
        return "exception_axis_deep_repeat_ge5"
    if event_count >= 3:
        return "exception_axis_moderate_repeat_3_to_4"
    return "exception_axis_thin_repeat_2"


def _exception_confidence_tier(event_count: int) -> str:
    if event_count >= 5:
        return "v2_2_exception_axis_deep_confidence"
    if event_count >= 3:
        return "v2_2_exception_axis_moderate_confidence"
    return "v2_2_exception_axis_thin_confidence"


def _exception_support_depth_read(event_count: int) -> str:
    if event_count >= 5:
        return "exception-axis recovered coherent subfamily with deeper repeated support"
    if event_count >= 3:
        return "exception-axis recovered coherent subfamily with moderate repeated support"
    return "exception-axis recovered coherent subfamily with minimal repeated support"


def _existing_v2_2_rule_status(axis_rule_status: str) -> str:
    if axis_rule_status == "primary_axis_retained_with_secondary_axis_gain":
        return "v2_2_retained_primary_axis_caveat"
    return "v2_2_inherited_v2_1_primitive"


def _existing_v2_2_read(axis_rule_status: str) -> str:
    if axis_rule_status == "primary_axis_retained_with_secondary_axis_gain":
        return "retained from v2.1 with the existing secondary-axis caveat"
    return "retained unchanged from the v2.1 primitive registry"


def _prepare_existing_registry(v2_1_registry: pd.DataFrame) -> pd.DataFrame:
    rows = v2_1_registry.copy()
    rows["definition_core_v2_2_status"] = "definition_core_v2_2_retained_v2_1_primitive"
    rows["definition_core_v2_2_rule_status"] = rows["axis_rule_status"].map(
        _existing_v2_2_rule_status
    )
    rows["definition_core_v2_2_read"] = rows["axis_rule_status"].map(_existing_v2_2_read)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _exception_primitives(
    *,
    recovered_subfamilies: pd.DataFrame,
    axis_exception_ledger: pd.DataFrame,
) -> pd.DataFrame:
    family_vector_lookup = {
        str(row["family_id"]): str(row.get("family_vector_class", ""))
        for _, row in axis_exception_ledger.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, source in recovered_subfamilies.iterrows():
        event_count = int(source["event_count"])
        row = {
            "primitive_id": source["candidate_subfamily_id"],
            "primitive_type": "exception_axis_recovered_subfamily",
            "definition_core_v2_1_status": "outside_v2_1_strong_axis_exception_candidate",
            "definition_core_v2_2_status": "definition_core_v2_2_added_exception_axis_primitive",
            "definition_confidence_tier": _exception_confidence_tier(event_count),
            "support_depth_tier": _exception_support_depth_tier(event_count),
            "support_confidence_tier": _exception_confidence_tier(event_count),
            "axis_rule_status": "exception_axis_promoted_from_strong_candidate",
            "definition_core_v2_2_rule_status": "v2_2_exception_axis_rule_added",
            "definition_core_v2_1_read": (
                "not in v2.1; strong axis exception remained outside primitive registry"
            ),
            "definition_core_v2_2_read": (
                "added by explicit exception-axis rule from strong axis-exception ledger"
            ),
            "support_depth_read": _exception_support_depth_read(event_count),
            "source_family_id": source["family_id"],
            "source_definition_core_v1_status": source["definition_core_v1_status"],
            "primitive_vector_class": source["candidate_subfamily_vector_class"],
            "primitive_coherence_status": source["candidate_subfamily_coherence_status"],
            "branch": source["branch"],
            "boundary_family_tier": source["boundary_family_tier"],
            "event_count": event_count,
            "source_event_count": int(source["source_event_count"]),
            "event_count_share_of_source_family": float(source["event_count_share_of_target"]),
            "decomposition_axis": source["candidate_axis"],
            "decomposition_key": source["candidate_key"],
            "dominant_split_vector_class": source["dominant_split_vector_class"],
            "dominant_host_context_class": source["dominant_host_context_class"],
            "dominant_shape_core_signature": source["dominant_shape_core_signature"],
            "dominant_host_handle_id": source["dominant_host_handle_id"],
            "route_execution_status": ROUTE_EXECUTION_STATUS,
            "wall_promotion_status": WALL_PROMOTION_STATUS,
            "quality_cost_status": QUALITY_COST_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
            "primitive_origin": "promoted_from_v2_1_strong_exception_axis_candidate",
            "definition_core_v2_status": "outside_v2_primary_axis_registry",
            "definition_core_v2_read": (
                "not promoted by primary-axis v2 rule; recovered by explicit v2.2 "
                "exception-axis rule"
            ),
            "source_family_vector_class": family_vector_lookup.get(str(source["family_id"]), ""),
            "ref_cluster_id": _ref_cluster_id(str(source["family_id"])),
            "definition_readiness": "definition_core_exception_axis",
            "source_ref_unit_count": pd.NA,
            "source_ref_weight_sum": pd.NA,
            "decomposition_axis_columns": source["candidate_axis_columns"],
            "dominant_split_vector_class_share": source["dominant_split_vector_class_share"],
            "dominant_host_context_class_share": source["dominant_host_context_class_share"],
            "dominant_shape_core_signature_share": source[
                "dominant_shape_core_signature_share"
            ],
            "dominant_host_handle_share": source["dominant_host_handle_share"],
            "top1_segment_share_median": source["top1_segment_share_median"],
            "top1_segment_share_iqr": source["top1_segment_share_iqr"],
            "top2_segment_share_median": source["top2_segment_share_median"],
            "top2_segment_share_iqr": source["top2_segment_share_iqr"],
            "effective_segment_count_median": source["effective_segment_count_median"],
            "effective_segment_count_iqr": source["effective_segment_count_iqr"],
            "split_vector_class_counts": source["split_vector_class_counts"],
            "host_context_class_counts": source["host_context_class_counts"],
            "boundary_pattern_counts": source["boundary_pattern_counts"],
            "exception_axis_target_unit_id": source["target_unit_id"],
            "exception_axis_candidate_axis": source["candidate_axis"],
            "exception_axis_candidate_axis_origin": source["candidate_axis_origin"],
            "exception_axis_candidate_key": source["candidate_key"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _combine_registry(
    *,
    existing_registry: pd.DataFrame,
    exception_primitives: pd.DataFrame,
) -> pd.DataFrame:
    rows = pd.concat([existing_registry, exception_primitives], ignore_index=True, sort=False)
    if rows["primitive_id"].duplicated().any():
        duplicates = rows.loc[rows["primitive_id"].duplicated(), "primitive_id"].tolist()
        raise ValueError(f"duplicate primitive ids: {duplicates[:5]}")
    preferred = [
        "primitive_id",
        "primitive_type",
        "definition_core_v2_2_status",
        "definition_core_v2_2_rule_status",
        "definition_confidence_tier",
        "support_depth_tier",
        "support_confidence_tier",
        "axis_rule_status",
        "definition_core_v2_2_read",
        "definition_core_v2_1_status",
        "definition_core_v2_1_read",
        "support_depth_read",
        "source_family_id",
        "source_definition_core_v1_status",
        "primitive_vector_class",
        "primitive_coherence_status",
        "branch",
        "boundary_family_tier",
        "event_count",
        "source_event_count",
        "event_count_share_of_source_family",
        "decomposition_axis",
        "decomposition_key",
        "dominant_split_vector_class",
        "dominant_host_context_class",
        "dominant_shape_core_signature",
        "dominant_host_handle_id",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows.loc[:, preferred + remainder].sort_values(
        [
            "definition_core_v2_2_status",
            "definition_confidence_tier",
            "primitive_type",
            "boundary_family_tier",
            "event_count",
            "primitive_id",
        ],
        ascending=[True, True, True, True, False, True],
    )


def _prepare_existing_events(
    *,
    v2_1_event_rows: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    meta_columns = [
        "primitive_id",
        "definition_core_v2_2_status",
        "definition_core_v2_2_rule_status",
        "definition_core_v2_2_read",
    ]
    rows = v2_1_event_rows.merge(
        registry[meta_columns],
        on="primitive_id",
        how="left",
        validate="many_to_one",
    )
    if rows["definition_core_v2_2_status"].isna().any():
        raise ValueError("missing v2.2 metadata for inherited event rows")
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _exception_events(
    *,
    recovered_events: pd.DataFrame,
    exception_primitives: pd.DataFrame,
    recovered_subfamilies: pd.DataFrame,
) -> pd.DataFrame:
    primitive_meta = exception_primitives[
        [
            "primitive_id",
            "primitive_type",
            "definition_core_v2_2_status",
            "definition_core_v2_2_rule_status",
            "definition_confidence_tier",
            "support_depth_tier",
            "support_confidence_tier",
            "axis_rule_status",
            "definition_core_v2_2_read",
            "source_family_id",
            "source_definition_core_v1_status",
            "primitive_vector_class",
            "primitive_coherence_status",
        ]
    ]
    known_subfamilies = set(recovered_subfamilies["candidate_subfamily_id"].astype(str))
    event_subfamilies = set(recovered_events["candidate_subfamily_id"].astype(str))
    missing_subfamilies = sorted(event_subfamilies - known_subfamilies)
    if missing_subfamilies:
        raise ValueError(
            "missing recovered subfamily metadata for exception-axis events: "
            f"{missing_subfamilies[:5]}"
        )
    rows = recovered_events.rename(columns={"candidate_subfamily_id": "primitive_id"}).copy()
    rows = rows.merge(primitive_meta, on="primitive_id", how="left", validate="many_to_one")
    if rows["definition_core_v2_2_status"].isna().any():
        raise ValueError("missing v2.2 metadata for exception-axis event rows")
    rows["definition_core_v2_1_status"] = "outside_v2_1_strong_axis_exception_candidate"
    rows["definition_core_v2_status"] = "outside_v2_primary_axis_registry"
    rows["definition_core_v2_event_scope"] = "exception_axis_primitive_event"
    rows["definition_core_v2_1_read"] = (
        "not in v2.1; strong axis exception remained outside primitive registry"
    )
    rows["definition_refinement_result"] = "exception_axis_recovered_coherent_subfamily"
    rows["subfamily_coherence_status"] = rows["candidate_subfamily_coherence_status"]
    rows["source_family_id"] = rows["source_family_id"].fillna(rows["family_id"])
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _combine_event_rows(
    *,
    existing_events: pd.DataFrame,
    exception_events: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    rows = pd.concat([existing_events, exception_events], ignore_index=True, sort=False)
    if rows["event_id"].duplicated().any():
        duplicates = rows.loc[rows["event_id"].duplicated(), "event_id"].tolist()
        raise ValueError(f"duplicate event ids across primitive rows: {duplicates[:5]}")
    expected = int(registry["event_count"].sum())
    if len(rows) != expected:
        raise ValueError(f"primitive event count mismatch: {len(rows)} != {expected}")
    preferred = [
        "primitive_id",
        "primitive_type",
        "definition_core_v2_2_status",
        "definition_core_v2_2_rule_status",
        "definition_confidence_tier",
        "support_depth_tier",
        "axis_rule_status",
        "source_family_id",
        "event_id",
        "branch",
        "boundary_family_tier",
        "split_vector_class",
        "host_context_class",
        "shape_core_signature",
        "comparison_seed",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows.loc[:, preferred + remainder]


def _axis_exception_ledger_v2_2(
    *,
    v2_1_axis_exception_ledger: pd.DataFrame,
    strong_targets: pd.DataFrame,
) -> pd.DataFrame:
    target_meta = strong_targets.set_index("family_id")[
        [
            "candidate_axis",
            "candidate_recovered_subfamily_count",
            "candidate_recovered_event_count",
            "candidate_recovered_event_share",
            "candidate_tiny_subfamily_count",
            "candidate_tiny_event_count",
            "candidate_unresolved_event_count",
        ]
    ].to_dict("index")
    rows = v2_1_axis_exception_ledger.copy()
    statuses: list[str] = []
    effects: list[str] = []
    actions: list[str] = []
    promoted_subfamilies: list[int] = []
    promoted_events: list[int] = []
    tiny_events: list[int] = []
    unresolved_events: list[int] = []
    for _, row in rows.iterrows():
        family_id = str(row["family_id"])
        old_status = str(row["definition_core_v2_1_exception_status"])
        meta = target_meta.get(family_id)
        if old_status == "strong_axis_exception_candidate_not_promoted" and meta is not None:
            statuses.append("strong_exception_axis_promoted_to_definition_primitive")
            effects.append("exception_axis_primitives_added_to_v2_2")
            actions.append("use_exception_axis_primitive_with_tiny_holdout_caveat")
            promoted_subfamilies.append(int(meta["candidate_recovered_subfamily_count"]))
            promoted_events.append(int(meta["candidate_recovered_event_count"]))
            tiny_events.append(int(meta["candidate_tiny_event_count"]))
            unresolved_events.append(int(meta["candidate_unresolved_event_count"]))
        elif old_status == "weak_axis_exception_diagnostic_not_promoted":
            statuses.append("weak_axis_exception_diagnostic_still_not_promoted")
            effects.append("outside_v2_2_primitive_registry")
            actions.append("hold_as_diagnostic_until_more_support_or_rule_edge_review")
            promoted_subfamilies.append(0)
            promoted_events.append(0)
            tiny_events.append(0)
            unresolved_events.append(int(row["source_event_count"]))
        else:
            statuses.append("marginal_secondary_axis_gain_primary_retained")
            effects.append("existing_primary_primitive_retained_with_axis_caveat")
            actions.append("retain_primary_axis_and_record_best_axis_as_secondary_check")
            promoted_subfamilies.append(0)
            promoted_events.append(0)
            tiny_events.append(0)
            unresolved_events.append(0)
    rows["definition_core_v2_2_exception_status"] = statuses
    rows["definition_core_v2_2_registry_effect"] = effects
    rows["next_definition_action"] = actions
    rows["v2_2_promoted_exception_axis_subfamily_count"] = promoted_subfamilies
    rows["v2_2_promoted_exception_axis_event_count"] = promoted_events
    rows["v2_2_exception_axis_tiny_holdout_event_count"] = tiny_events
    rows["v2_2_exception_axis_unresolved_event_count"] = unresolved_events
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "family_id",
        "definition_core_v1_status",
        "definition_core_v2_1_exception_status",
        "definition_core_v2_2_exception_status",
        "definition_core_v2_2_registry_effect",
        "next_definition_action",
        "source_event_count",
        "primary_axis",
        "best_axis",
        "primary_recovered_event_count",
        "best_recovered_event_count",
        "best_recovered_event_share",
        "v2_2_promoted_exception_axis_subfamily_count",
        "v2_2_promoted_exception_axis_event_count",
        "v2_2_exception_axis_tiny_holdout_event_count",
        "v2_2_exception_axis_unresolved_event_count",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows.loc[:, preferred + remainder].sort_values(
        [
            "definition_core_v2_2_exception_status",
            "v2_2_promoted_exception_axis_event_count",
            "family_id",
        ],
        ascending=[True, False, True],
    )


def _exception_tiny_holdouts(tiny_subfamilies: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in tiny_subfamilies.iterrows():
        rows.append(
            {
                "audit_id": source["candidate_subfamily_id"],
                "source_family_id": source["family_id"],
                "source_definition_core_v1_status": source["definition_core_v1_status"],
                "definition_core_v2_1_queue_status": source["source_queue_status"],
                "definition_core_v2_2_queue_status": "exception_axis_tiny_holdout",
                "definition_core_v2_audit_status": "tiny_exception_axis_holdout",
                "residual_definition_read": (
                    "exception_axis_single_event_or_tiny_support_do_not_promote"
                ),
                "definition_core_v2_2_queue_read": (
                    "strong exception-axis rule recovered the family, but this "
                    "subfamily has only singleton support"
                ),
                "event_count": int(source["event_count"]),
                "decomposition_axis": source["candidate_axis"],
                "decomposition_key": source["candidate_key"],
                "subfamily_coherence_status": source[
                    "candidate_subfamily_coherence_status"
                ],
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "audit_row_type": "exception_axis_tiny_subfamily_not_promoted",
                "event_count_basis": "exception_axis_candidate_subfamily_event_count",
                "boundary_family_tier": source["boundary_family_tier"],
                "decomposition_axis_columns": source["candidate_axis_columns"],
                "definition_refinement_result": source["candidate_definition_result"],
                "primary_axis": source["primary_axis"],
                "best_axis": source["candidate_axis"],
                "primary_recovered_coherent_event_count": 0,
                "best_axis_recovered_coherent_event_count": 0,
                "family_refinement_read": "tiny holdout after v2.2 exception-axis rule",
            }
        )
    return pd.DataFrame(rows)


def _residual_queue_v2_2(
    *,
    v2_1_residual_queue: pd.DataFrame,
    strong_family_ids: set[str],
    tiny_holdouts: pd.DataFrame,
) -> pd.DataFrame:
    inherited = v2_1_residual_queue[
        ~v2_1_residual_queue["source_family_id"].astype(str).isin(strong_family_ids)
    ].copy()
    inherited["definition_core_v2_2_queue_status"] = inherited[
        "definition_core_v2_1_queue_status"
    ]
    inherited["definition_core_v2_2_queue_read"] = "inherited unresolved residual from v2.1"
    inherited["route_execution_status"] = ROUTE_EXECUTION_STATUS
    inherited["wall_promotion_status"] = WALL_PROMOTION_STATUS
    inherited["quality_cost_status"] = QUALITY_COST_STATUS
    inherited["claim_boundary"] = CLAIM_BOUNDARY
    rows = pd.concat([inherited, tiny_holdouts], ignore_index=True, sort=False)
    preferred = [
        "audit_id",
        "source_family_id",
        "source_definition_core_v1_status",
        "definition_core_v2_2_queue_status",
        "definition_core_v2_1_queue_status",
        "definition_core_v2_2_queue_read",
        "definition_core_v2_audit_status",
        "residual_definition_read",
        "event_count",
        "decomposition_axis",
        "decomposition_key",
        "subfamily_coherence_status",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows.loc[:, preferred + remainder].sort_values(
        [
            "definition_core_v2_2_queue_status",
            "source_definition_core_v1_status",
            "event_count",
            "audit_id",
        ],
        ascending=[True, True, False, True],
    )


def _confidence_summary(registry: pd.DataFrame) -> pd.DataFrame:
    rows = (
        registry.groupby(
            [
                "primitive_type",
                "definition_core_v2_2_rule_status",
                "definition_confidence_tier",
                "support_depth_tier",
                "axis_rule_status",
            ],
            as_index=False,
        )
        .agg(
            primitive_count=("primitive_id", "size"),
            event_count_sum=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
            median_event_count=("event_count", "median"),
        )
        .sort_values(
            ["primitive_type", "definition_core_v2_2_rule_status", "event_count_sum"],
            ascending=[True, True, False],
        )
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _axis_rule_summary(
    *,
    registry: pd.DataFrame,
    axis_exception_ledger: pd.DataFrame,
    residual_queue: pd.DataFrame,
) -> pd.DataFrame:
    primitive_rows = (
        registry.groupby(["definition_core_v2_2_rule_status"], as_index=False)
        .agg(
            row_count=("primitive_id", "size"),
            event_count_sum=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
        )
        .rename(columns={"definition_core_v2_2_rule_status": "axis_rule_or_queue_status"})
    )
    primitive_rows["summary_scope"] = "primitive_v2_2_rule_status"
    exception_rows = (
        axis_exception_ledger.groupby(["definition_core_v2_2_exception_status"], as_index=False)
        .agg(
            row_count=("family_id", "size"),
            event_count_sum=("source_event_count", "sum"),
            source_family_count=("family_id", "nunique"),
        )
        .rename(columns={"definition_core_v2_2_exception_status": "axis_rule_or_queue_status"})
    )
    exception_rows["summary_scope"] = "axis_exception_ledger_v2_2_status"
    residual_rows = (
        residual_queue.groupby(["definition_core_v2_2_queue_status"], as_index=False)
        .agg(
            row_count=("audit_id", "size"),
            event_count_sum=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
        )
        .rename(columns={"definition_core_v2_2_queue_status": "axis_rule_or_queue_status"})
    )
    residual_rows["summary_scope"] = "residual_definition_queue_v2_2_status"
    rows = pd.concat([primitive_rows, exception_rows, residual_rows], ignore_index=True)
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(["summary_scope", "event_count_sum"], ascending=[True, False])


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
    registry: pd.DataFrame,
    exception_primitives: pd.DataFrame,
    tiny_holdouts: pd.DataFrame,
    axis_exception_ledger: pd.DataFrame,
    residual_queue: pd.DataFrame,
    confidence_summary: pd.DataFrame,
    axis_rule_summary: pd.DataFrame,
    v2_1_primitive_event_count: int,
    v2_1_residual_event_count: int,
) -> None:
    text = [
        "# NanoClustering Definition-Core V2.2 Exception-Axis Registry",
        "",
        f"- primitive_rows: `{len(registry)}`",
        f"- primitive_event_count: `{int(registry['event_count'].sum())}`",
        f"- source_family_count: `{registry['source_family_id'].nunique()}`",
        f"- added_exception_axis_primitives: `{len(exception_primitives)}`",
        f"- added_exception_axis_events: `{int(exception_primitives['event_count'].sum())}`",
        f"- exception_axis_tiny_holdout_events: `{int(tiny_holdouts['event_count'].sum()) if not tiny_holdouts.empty else 0}`",
        f"- residual_definition_queue_events: `{int(residual_queue['event_count'].sum())}`",
        f"- v2_1_primitive_event_count: `{v2_1_primitive_event_count}`",
        f"- v2_1_residual_definition_queue_events: `{v2_1_residual_event_count}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Added Exception-Axis Primitives",
        "",
        _markdown_table(
            exception_primitives,
            [
                "primitive_id",
                "source_family_id",
                "decomposition_axis",
                "event_count",
                "definition_confidence_tier",
                "dominant_split_vector_class",
                "dominant_host_context_class",
                "dominant_shape_core_signature",
            ],
            max_rows=20,
        ),
        "",
        "## Axis Exception Ledger",
        "",
        _markdown_table(
            axis_exception_ledger,
            [
                "family_id",
                "definition_core_v2_1_exception_status",
                "definition_core_v2_2_exception_status",
                "source_event_count",
                "best_axis",
                "v2_2_promoted_exception_axis_event_count",
                "v2_2_exception_axis_tiny_holdout_event_count",
            ],
            max_rows=30,
        ),
        "",
        "## Confidence Summary",
        "",
        _markdown_table(
            confidence_summary,
            [
                "primitive_type",
                "definition_core_v2_2_rule_status",
                "definition_confidence_tier",
                "support_depth_tier",
                "primitive_count",
                "event_count_sum",
                "source_family_count",
            ],
            max_rows=60,
        ),
        "",
        "## Axis And Queue Summary",
        "",
        _markdown_table(
            axis_rule_summary,
            [
                "summary_scope",
                "axis_rule_or_queue_status",
                "row_count",
                "event_count_sum",
                "source_family_count",
            ],
            max_rows=60,
        ),
        "",
        "## Residual Definition Queue",
        "",
        _markdown_table(
            residual_queue.groupby(
                [
                    "definition_core_v2_2_queue_status",
                    "source_definition_core_v1_status",
                    "residual_definition_read",
                ],
                as_index=False,
            )
            .agg(
                queue_row_count=("audit_id", "size"),
                event_count_sum=("event_count", "sum"),
                source_family_count=("source_family_id", "nunique"),
            )
            .sort_values(["definition_core_v2_2_queue_status", "event_count_sum"], ascending=[True, False]),
            [
                "definition_core_v2_2_queue_status",
                "source_definition_core_v1_status",
                "residual_definition_read",
                "queue_row_count",
                "event_count_sum",
                "source_family_count",
            ],
            max_rows=40,
        ),
        "",
        "## Read",
        "",
        "- V2.2 adds only strong exception-axis recovered coherent subfamilies; second-axis and joint-axis candidates remain outside the primitive registry.",
        "- Strong exception-axis coverage is now encoded as primitives for 24 of 29 events, while 5 singleton/tiny events remain explicit holdouts.",
        "- Primitive event coverage rises from 886 to 910 events, and the residual definition queue falls from 140 to 116 events.",
        "- This is still membership-derived basin-definition cartography, not a final global attraction-basin, wall/pathway, quality, cost, or directed-search result.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def _validate_inputs(
    *,
    v2_1_event_rows: pd.DataFrame,
    strong_targets: pd.DataFrame,
    recovered_subfamilies: pd.DataFrame,
    recovered_events: pd.DataFrame,
    tiny_subfamilies: pd.DataFrame,
) -> None:
    if len(strong_targets) != 4:
        raise ValueError(f"expected 4 strong exception-axis target rows, found {len(strong_targets)}")
    if not strong_targets["candidate_axis_read"].eq("candidate_axis_recovers_most_events").all():
        raise ValueError("all strong exception-axis target rows must recover most events")
    if len(recovered_subfamilies) != 8:
        raise ValueError(
            f"expected 8 recovered strong exception-axis subfamilies, found {len(recovered_subfamilies)}"
        )
    if int(recovered_subfamilies["event_count"].sum()) != 24:
        raise ValueError("strong exception-axis recovered subfamilies must cover 24 events")
    if len(recovered_events) != 24 or recovered_events["event_id"].nunique() != 24:
        raise ValueError("strong exception-axis recovered event rows must contain 24 unique events")
    if int(tiny_subfamilies["event_count"].sum()) != 5:
        raise ValueError("strong exception-axis tiny holdouts must cover 5 events")
    overlap = set(recovered_events["event_id"]).intersection(set(v2_1_event_rows["event_id"]))
    if overlap:
        raise ValueError(f"exception-axis recovered events overlap v2.1 primitives: {sorted(overlap)[:5]}")


def materialize(
    *,
    v2_1_registry_dir: Path,
    axis_rule_candidates_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    v2_1_registry = _read_csv(v2_1_registry_dir / V2_1_PRIMITIVE_REGISTRY_CSV)
    v2_1_event_rows = _read_csv(v2_1_registry_dir / V2_1_PRIMITIVE_EVENT_ROWS_CSV)
    v2_1_axis_exception_ledger = _read_csv(
        v2_1_registry_dir / V2_1_AXIS_EXCEPTION_LEDGER_CSV
    )
    v2_1_residual_queue = _read_csv(
        v2_1_registry_dir / V2_1_RESIDUAL_DEFINITION_QUEUE_CSV
    )
    target_rows = _read_csv(axis_rule_candidates_dir / AXIS_RULE_TARGET_ROWS_CSV)
    candidate_subfamilies = _read_csv(axis_rule_candidates_dir / AXIS_RULE_SUBFAMILY_ROWS_CSV)
    candidate_events = _read_csv(axis_rule_candidates_dir / AXIS_RULE_EVENT_ROWS_CSV)

    strong_targets = target_rows[target_rows["rule_scope"].eq(STRONG_RULE_SCOPE)].copy()
    strong_subfamilies = candidate_subfamilies[
        candidate_subfamilies["rule_scope"].eq(STRONG_RULE_SCOPE)
    ].copy()
    recovered_subfamilies = strong_subfamilies[
        strong_subfamilies["candidate_definition_result"].eq(RECOVERED_RESULT)
    ].copy()
    tiny_subfamilies = strong_subfamilies[
        strong_subfamilies["candidate_definition_result"].eq(TINY_RESULT)
    ].copy()
    recovered_events = candidate_events[
        candidate_events["rule_scope"].eq(STRONG_RULE_SCOPE)
        & candidate_events["candidate_definition_result"].eq(RECOVERED_RESULT)
    ].copy()

    _validate_inputs(
        v2_1_event_rows=v2_1_event_rows,
        strong_targets=strong_targets,
        recovered_subfamilies=recovered_subfamilies,
        recovered_events=recovered_events,
        tiny_subfamilies=tiny_subfamilies,
    )

    existing_registry = _prepare_existing_registry(v2_1_registry)
    exception_primitives = _exception_primitives(
        recovered_subfamilies=recovered_subfamilies,
        axis_exception_ledger=v2_1_axis_exception_ledger,
    )
    registry = _combine_registry(
        existing_registry=existing_registry,
        exception_primitives=exception_primitives,
    )
    existing_events = _prepare_existing_events(
        v2_1_event_rows=v2_1_event_rows,
        registry=registry,
    )
    added_events = _exception_events(
        recovered_events=recovered_events,
        exception_primitives=exception_primitives,
        recovered_subfamilies=recovered_subfamilies,
    )
    event_rows = _combine_event_rows(
        existing_events=existing_events,
        exception_events=added_events,
        registry=registry,
    )
    axis_exception_ledger = _axis_exception_ledger_v2_2(
        v2_1_axis_exception_ledger=v2_1_axis_exception_ledger,
        strong_targets=strong_targets,
    )
    tiny_holdouts = _exception_tiny_holdouts(tiny_subfamilies)
    residual_queue = _residual_queue_v2_2(
        v2_1_residual_queue=v2_1_residual_queue,
        strong_family_ids=set(strong_targets["family_id"].astype(str)),
        tiny_holdouts=tiny_holdouts,
    )
    confidence_summary = _confidence_summary(registry)
    axis_rule_summary = _axis_rule_summary(
        registry=registry,
        axis_exception_ledger=axis_exception_ledger,
        residual_queue=residual_queue,
    )

    v2_1_primitive_event_count = int(v2_1_registry["event_count"].sum())
    v2_1_residual_event_count = int(v2_1_residual_queue["event_count"].sum())
    if len(registry) != len(v2_1_registry) + len(exception_primitives):
        raise ValueError("v2.2 primitive count mismatch")
    if int(registry["event_count"].sum()) != v2_1_primitive_event_count + 24:
        raise ValueError("v2.2 primitive event count must add exactly 24 events")
    if int(residual_queue["event_count"].sum()) != v2_1_residual_event_count - 24:
        raise ValueError("v2.2 residual event count must remove exactly 24 promoted events")
    if int(registry["event_count"].sum()) + int(residual_queue["event_count"].sum()) != (
        v2_1_primitive_event_count + v2_1_residual_event_count
    ):
        raise ValueError("v2.2 primitive plus residual events must preserve v2.1 universe size")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(registry, output_dir / V2_2_PRIMITIVE_REGISTRY_CSV)
    _write_csv(event_rows, output_dir / V2_2_PRIMITIVE_EVENT_ROWS_CSV)
    _write_csv(exception_primitives, output_dir / V2_2_EXCEPTION_AXIS_PRIMITIVE_ROWS_CSV)
    _write_csv(tiny_holdouts, output_dir / V2_2_EXCEPTION_AXIS_TINY_HOLDOUT_ROWS_CSV)
    _write_csv(axis_exception_ledger, output_dir / V2_2_AXIS_EXCEPTION_LEDGER_CSV)
    _write_csv(residual_queue, output_dir / V2_2_RESIDUAL_DEFINITION_QUEUE_CSV)
    _write_csv(confidence_summary, output_dir / V2_2_CONFIDENCE_SUMMARY_CSV)
    _write_csv(axis_rule_summary, output_dir / V2_2_AXIS_RULE_SUMMARY_CSV)
    _write_report(
        output_dir=output_dir,
        registry=registry,
        exception_primitives=exception_primitives,
        tiny_holdouts=tiny_holdouts,
        axis_exception_ledger=axis_exception_ledger,
        residual_queue=residual_queue,
        confidence_summary=confidence_summary,
        axis_rule_summary=axis_rule_summary,
        v2_1_primitive_event_count=v2_1_primitive_event_count,
        v2_1_residual_event_count=v2_1_residual_event_count,
    )

    summary = {
        "ok": True,
        "v2_1_registry_dir": _rel(v2_1_registry_dir),
        "axis_rule_candidates_dir": _rel(axis_rule_candidates_dir),
        "output_dir": _rel(output_dir),
        "primitive_row_count": int(len(registry)),
        "primitive_event_row_count": int(len(event_rows)),
        "primitive_event_count_sum": int(registry["event_count"].sum()),
        "primitive_source_family_count": int(registry["source_family_id"].nunique()),
        "v2_1_primitive_event_count": v2_1_primitive_event_count,
        "added_exception_axis_primitive_count": int(len(exception_primitives)),
        "added_exception_axis_event_count": int(exception_primitives["event_count"].sum()),
        "added_exception_axis_source_family_count": int(
            exception_primitives["source_family_id"].nunique()
        ),
        "exception_axis_tiny_holdout_event_count": int(tiny_holdouts["event_count"].sum())
        if not tiny_holdouts.empty
        else 0,
        "residual_definition_queue_row_count": int(len(residual_queue)),
        "residual_definition_queue_event_count": int(residual_queue["event_count"].sum()),
        "v2_1_residual_definition_queue_event_count": v2_1_residual_event_count,
        "definition_confidence_tier_counts": _count(registry, "definition_confidence_tier"),
        "definition_core_v2_2_rule_status_counts": _count(
            registry,
            "definition_core_v2_2_rule_status",
        ),
        "axis_exception_v2_2_status_counts": _count(
            axis_exception_ledger,
            "definition_core_v2_2_exception_status",
        ),
        "residual_definition_queue_v2_2_status_counts": _count(
            residual_queue,
            "definition_core_v2_2_queue_status",
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "primitive_registry_csv": _rel(output_dir / V2_2_PRIMITIVE_REGISTRY_CSV),
            "primitive_event_rows_csv": _rel(output_dir / V2_2_PRIMITIVE_EVENT_ROWS_CSV),
            "exception_axis_primitive_rows_csv": _rel(
                output_dir / V2_2_EXCEPTION_AXIS_PRIMITIVE_ROWS_CSV
            ),
            "exception_axis_tiny_holdout_rows_csv": _rel(
                output_dir / V2_2_EXCEPTION_AXIS_TINY_HOLDOUT_ROWS_CSV
            ),
            "axis_exception_ledger_csv": _rel(output_dir / V2_2_AXIS_EXCEPTION_LEDGER_CSV),
            "residual_definition_queue_csv": _rel(
                output_dir / V2_2_RESIDUAL_DEFINITION_QUEUE_CSV
            ),
            "confidence_summary_csv": _rel(output_dir / V2_2_CONFIDENCE_SUMMARY_CSV),
            "axis_rule_summary_csv": _rel(output_dir / V2_2_AXIS_RULE_SUMMARY_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "v2_1_registry_dir": _rel(v2_1_registry_dir),
        "axis_rule_candidates_dir": _rel(axis_rule_candidates_dir),
        "output_dir": _rel(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
        "v2_2_rule": (
            "Inherit v2.1 primitives and add only strong exception-axis recovered "
            "coherent subfamilies; keep second-axis and joint-axis candidates out "
            "of the primitive registry."
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
    parser.add_argument("--v2-1-registry-dir", type=Path, default=DEFAULT_V2_1_REGISTRY_DIR)
    parser.add_argument(
        "--axis-rule-candidates-dir",
        type=Path,
        default=DEFAULT_AXIS_RULE_CANDIDATES_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        v2_1_registry_dir=args.v2_1_registry_dir.resolve(),
        axis_rule_candidates_dir=args.axis_rule_candidates_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
