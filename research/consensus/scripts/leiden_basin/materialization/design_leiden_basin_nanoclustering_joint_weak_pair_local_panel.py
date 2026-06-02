#!/usr/bin/env python3
"""Design a NanoClustering local panel for joint weak-pair analogs.

This script is a read-only design step after the NanoClustering joint weak-pair
analog screen. It freezes candidate/control cases, pre-endpoint role rows, and
the future endpoint-replay readout contract. It does not run clustering,
execute endpoint replay, construct route/pathway traces, promote walls, inspect
quality/cost, or claim real-data method success.
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
DEFAULT_ANALOG_SCREEN_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_joint_weak_pair_analog_screen_20260601"
)
DEFAULT_MEASUREMENT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_v2_2_measurement_panel_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_joint_weak_pair_local_panel_design_20260601"
)

ANALOG_PRIMITIVE_ROWS_CSV = "nanoclustering_joint_weak_pair_analog_primitive_rows.csv"
ANALOG_CONTROL_ROWS_CSV = "nanoclustering_joint_weak_pair_analog_matched_control_rows.csv"
MEASUREMENT_EVENT_ROWS_CSV = "nanoclustering_v2_2_accepted_primitive_event_measurement_rows.csv"

PANEL_CASE_ROWS_CSV = "nanoclustering_joint_weak_pair_local_panel_case_rows.csv"
PANEL_ROLE_ROWS_CSV = "nanoclustering_joint_weak_pair_local_panel_role_rows.csv"
PANEL_EVENT_ROLE_ROWS_CSV = "nanoclustering_joint_weak_pair_local_panel_event_role_rows.csv"
PANEL_ENDPOINT_SIGNATURE_ROWS_CSV = (
    "nanoclustering_joint_weak_pair_local_panel_endpoint_signature_rows.csv"
)
CONTROL_SENSITIVITY_ROWS_CSV = (
    "nanoclustering_joint_weak_pair_local_panel_control_sensitivity_rows.csv"
)
ENDPOINT_REPLAY_CONTRACT_CSV = (
    "nanoclustering_joint_weak_pair_local_panel_endpoint_replay_contract.csv"
)
GATE_MATRIX_CSV = "nanoclustering_joint_weak_pair_local_panel_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_joint_weak_pair_local_panel_summary.json"
CONFIG_JSON = "nanoclustering_joint_weak_pair_local_panel_config.json"
REPORT_MD = "nanoclustering_joint_weak_pair_local_panel_report.md"

TIER1 = "tier1_external_multi_fragment_host_competition_analog"
TIER2 = "tier2_recurrent_multi_fragment_analog"
STRICT_MATCHING_DISTANCE_MAX = 2.0

CLAIM_BOUNDARY = (
    "NanoClustering joint weak-pair local panel design only; freezes "
    "candidate/control cases, pre-endpoint roles, and endpoint-replay readout "
    "contract. It does not run clustering, execute endpoint replay, execute "
    "routes/pathways, promote walls, inspect quality/cost, or claim real-data "
    "method success."
)
ROLE_FREEZE_STATUS = "frozen_from_v2_2_measurement_and_analog_screen"
REPLAY_EXECUTION_STATUS = "not_executed_contract_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_local_panel_design"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


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


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["role_freeze_status"] = ROLE_FREEZE_STATUS
    rows["replay_execution_status"] = REPLAY_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _lookup_by_id(frame: pd.DataFrame, id_column: str) -> dict[str, pd.Series]:
    return {str(row[id_column]): row for _, row in frame.iterrows()}


def _num(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(row: pd.Series, column: str) -> str:
    value = row.get(column, "")
    if pd.isna(value):
        return ""
    return str(value)


def _panel_scope(analog_tier: str) -> str:
    if analog_tier == TIER1:
        return "core_tier1_panel"
    return "reserve_lower_tier_panel"


def _candidate_role(analog_tier: str) -> str:
    if analog_tier == TIER1:
        return "joint_external_multifragment_host_competition_candidate"
    if analog_tier == TIER2:
        return "recurrent_multifragment_reserve_candidate"
    return "lower_tier_multifragment_reserve_candidate"


def _control_role(control_tier: str, endpoint_host_scope: str) -> str:
    if control_tier == "source_preserved_multifragment_control_like":
        return "source_preserved_multifragment_control"
    if endpoint_host_scope == "all_events_source_host_preserved":
        return "matched_source_preserved_nonanalog_control"
    return "matched_nonanalog_control"


def _case_read(row: pd.Series) -> str:
    tier = _text(row, "candidate_analog_tier")
    if tier == TIER1:
        return (
            "core case: strongest external multi-fragment host-competition "
            "analog with matched non-analog control"
        )
    return "reserve case: lower-tier recurrent multi-fragment analog with matched control"


def _join_unique(values: pd.Series, *, max_items: int = 24) -> str:
    clean = sorted({str(value) for value in values.dropna() if str(value)})
    suffix = "" if len(clean) <= max_items else f";...(+{len(clean) - max_items})"
    return ";".join(clean[:max_items]) + suffix


def _join_counts(values: pd.Series) -> str:
    counts = values.dropna().astype(str).value_counts().sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items())


def _candidate_caveat_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if not bool(row["candidate_pure_external_host_scope"]):
        flags.append("mixed_or_source_preserved_candidate_host_scope")
    if not bool(row["candidate_good_reusable_control_match"]):
        flags.append("high_reusable_control_matching_distance")
    if not bool(row["candidate_no_residual_definition_debt"]):
        flags.append("candidate_residual_definition_debt")
    if int(row["reusable_control_reuse_count"]) >= 5:
        flags.append("high_reusable_control_anchor_reuse")
    return ";".join(flags) if flags else "none"


def _analysis_tier(row: pd.Series) -> str:
    if bool(row["strict_core_v0"]):
        return "strict_core_v0_primary"
    if row["panel_scope"] == "core_tier1_panel":
        return "full_core_caveated_sensitivity"
    return "reserve_exploratory"


def _annotate_case_rows(rows: pd.DataFrame) -> pd.DataFrame:
    annotated = rows.copy()
    annotated["reusable_control_reuse_count"] = annotated["control_primitive_id"].map(
        annotated["control_primitive_id"].value_counts()
    )
    annotated["candidate_pure_external_host_scope"] = annotated[
        "candidate_endpoint_host_scope"
    ].astype(str).eq("all_events_external_host_absorption")
    annotated["candidate_good_reusable_control_match"] = (
        pd.to_numeric(annotated["matching_distance"], errors="coerce")
        <= STRICT_MATCHING_DISTANCE_MAX
    )
    annotated["candidate_no_residual_definition_debt"] = annotated[
        "candidate_residual_caveat_status"
    ].astype(str).eq("source_family_has_no_residual_definition_debt")
    annotated["strict_core_v0"] = (
        annotated["panel_scope"].astype(str).eq("core_tier1_panel")
        & annotated["candidate_pure_external_host_scope"].astype(bool)
        & annotated["candidate_good_reusable_control_match"].astype(bool)
        & annotated["candidate_no_residual_definition_debt"].astype(bool)
    )
    annotated["analysis_tier"] = annotated.apply(_analysis_tier, axis=1)
    annotated["candidate_caveat_flags"] = annotated.apply(_candidate_caveat_flags, axis=1)
    annotated["endpoint_success_unit"] = "endpoint_family_signature_distance"
    annotated["single_endpoint_hit_allowed"] = False
    annotated["endpoint_success_readout"] = (
        "future replay must compare terminal endpoints to the frozen "
        "candidate event-family signature and matched control signature; a "
        "single top1 endpoint handle is not the success unit"
    )
    return annotated


def _build_case_rows(
    primitive_rows: pd.DataFrame,
    matched_controls: pd.DataFrame,
    *,
    max_cases: int | None,
) -> pd.DataFrame:
    primitives = _lookup_by_id(primitive_rows, "primitive_id")
    controls = matched_controls.copy()
    controls["_scope_order"] = controls["candidate_analog_tier"].map(
        {TIER1: 0, TIER2: 1}
    ).fillna(2)
    controls = controls.sort_values(
        [
            "_scope_order",
            "candidate_branch",
            "candidate_source_family_id",
            "candidate_primitive_id",
            "matching_distance",
        ],
        kind="mergesort",
    )
    if max_cases is not None:
        controls = controls.head(max_cases)

    rows: list[dict[str, Any]] = []
    for idx, control in enumerate(controls.itertuples(index=False), start=1):
        candidate_id = str(control.candidate_primitive_id)
        control_id = str(control.control_primitive_id)
        candidate = primitives[candidate_id]
        matched = primitives[control_id]
        scope = _panel_scope(str(control.candidate_analog_tier))
        branch = str(control.candidate_branch)
        case_id = f"jwpa_{idx:03d}_{_slug(candidate_id)[:64]}"
        rows.append(
            {
                "panel_case_id": case_id,
                "panel_case_rank": idx,
                "panel_scope": scope,
                "case_read": _case_read(pd.Series(control._asdict())),
                "candidate_primitive_id": candidate_id,
                "candidate_source_family_id": str(control.candidate_source_family_id),
                "candidate_branch": branch,
                "candidate_analog_tier": str(control.candidate_analog_tier),
                "candidate_pre_endpoint_role": _candidate_role(
                    str(control.candidate_analog_tier)
                ),
                "candidate_analog_score": int(_num(candidate, "analog_score")),
                "candidate_event_count": int(control.candidate_event_count),
                "candidate_split_vector_class_mode": _text(
                    candidate, "split_vector_class_mode"
                ),
                "candidate_host_context_class_mode": str(
                    control.candidate_host_context_class_mode
                ),
                "candidate_endpoint_host_scope": _text(candidate, "endpoint_host_scope"),
                "candidate_dominant_host_handle_id_mode": _text(
                    candidate, "dominant_host_handle_id_mode"
                ),
                "candidate_dominant_host_handle_id_mode_share": float(
                    control.candidate_dominant_host_handle_id_mode_share
                ),
                "candidate_effective_segment_count_median": float(
                    control.candidate_effective_segment_count_median
                ),
                "candidate_top1_segment_share_ref_weight_median": float(
                    control.candidate_top1_segment_share_ref_weight_median
                ),
                "candidate_top2_segment_share_ref_weight_median": _num(
                    candidate, "top2_segment_share_ref_weight_median"
                ),
                "candidate_split_segment_count_ge5_weight_median": _num(
                    candidate, "split_segment_count_ge5_weight_median"
                ),
                "candidate_residual_caveat_status": _text(
                    candidate, "residual_caveat_status"
                ),
                "control_primitive_id": control_id,
                "control_source_family_id": str(control.control_source_family_id),
                "control_branch": str(control.control_branch),
                "control_analog_tier": str(control.control_analog_tier),
                "control_pre_endpoint_role": _control_role(
                    str(control.control_analog_tier),
                    _text(matched, "endpoint_host_scope"),
                ),
                "control_event_count": int(control.control_event_count),
                "control_split_vector_class_mode": _text(matched, "split_vector_class_mode"),
                "control_host_context_class_mode": str(control.control_host_context_class_mode),
                "control_endpoint_host_scope": _text(matched, "endpoint_host_scope"),
                "control_effective_segment_count_median": float(
                    control.control_effective_segment_count_median
                ),
                "control_top1_segment_share_ref_weight_median": float(
                    control.control_top1_segment_share_ref_weight_median
                ),
                "matching_distance": float(control.matching_distance),
                "control_read": str(control.control_read),
                "pre_endpoint_freeze_basis": (
                    "candidate/control roles are frozen from analog-screen and "
                    "v2.2 measurement fields before any future endpoint replay"
                ),
                "future_endpoint_readout_status": "required_not_run",
            }
        )
    return _with_claim_columns(_annotate_case_rows(pd.DataFrame(rows)))


def _role_rows(case_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in case_rows.itertuples(index=False):
        rows.append(
            {
                "panel_case_id": case.panel_case_id,
                "panel_case_rank": case.panel_case_rank,
                "panel_scope": case.panel_scope,
                "analysis_tier": case.analysis_tier,
                "strict_core_v0": case.strict_core_v0,
                "role_id": f"{case.panel_case_id}__candidate",
                "role_side": "candidate",
                "primitive_id": case.candidate_primitive_id,
                "source_family_id": case.candidate_source_family_id,
                "branch": case.candidate_branch,
                "pre_endpoint_role": case.candidate_pre_endpoint_role,
                "analog_tier": case.candidate_analog_tier,
                "event_count": case.candidate_event_count,
                "split_vector_class_mode": case.candidate_split_vector_class_mode,
                "host_context_class_mode": case.candidate_host_context_class_mode,
                "endpoint_host_scope": case.candidate_endpoint_host_scope,
                "dominant_host_handle_id_mode": case.candidate_dominant_host_handle_id_mode,
                "dominant_host_handle_id_mode_share": case.candidate_dominant_host_handle_id_mode_share,
                "effective_segment_count_median": case.candidate_effective_segment_count_median,
                "top1_segment_share_ref_weight_median": case.candidate_top1_segment_share_ref_weight_median,
                "candidate_caveat_flags": case.candidate_caveat_flags,
                "endpoint_success_unit": case.endpoint_success_unit,
                "role_read": "candidate role frozen before future endpoint replay",
            }
        )
        rows.append(
            {
                "panel_case_id": case.panel_case_id,
                "panel_case_rank": case.panel_case_rank,
                "panel_scope": case.panel_scope,
                "analysis_tier": case.analysis_tier,
                "strict_core_v0": case.strict_core_v0,
                "role_id": f"{case.panel_case_id}__control",
                "role_side": "control",
                "primitive_id": case.control_primitive_id,
                "source_family_id": case.control_source_family_id,
                "branch": case.control_branch,
                "pre_endpoint_role": case.control_pre_endpoint_role,
                "analog_tier": case.control_analog_tier,
                "event_count": case.control_event_count,
                "split_vector_class_mode": case.control_split_vector_class_mode,
                "host_context_class_mode": case.control_host_context_class_mode,
                "endpoint_host_scope": case.control_endpoint_host_scope,
                "dominant_host_handle_id_mode": "",
                "dominant_host_handle_id_mode_share": None,
                "effective_segment_count_median": case.control_effective_segment_count_median,
                "top1_segment_share_ref_weight_median": case.control_top1_segment_share_ref_weight_median,
                "candidate_caveat_flags": case.candidate_caveat_flags,
                "endpoint_success_unit": case.endpoint_success_unit,
                "role_read": "matched control role frozen before future endpoint replay",
            }
        )
    return _with_claim_columns(pd.DataFrame(rows))


def _event_pre_endpoint_role(role_side: str, host_context: str, split_class: str) -> str:
    if role_side == "candidate" and host_context == "external_host_absorption":
        return "candidate_external_host_fragment_event"
    if role_side == "candidate" and host_context == "source_host_preserved":
        return "candidate_source_preserved_fragment_event"
    if role_side == "candidate":
        return "candidate_other_fragment_event"
    if role_side == "control" and host_context == "source_host_preserved":
        return "control_source_preserved_event"
    if role_side == "control" and host_context == "external_host_absorption":
        return "control_external_host_event"
    if "fragmentation" in split_class or "split" in split_class:
        return "control_fragment_event"
    return "control_other_event"


def _event_role_rows(case_rows: pd.DataFrame, event_rows: pd.DataFrame) -> pd.DataFrame:
    event_rows = event_rows.copy()
    event_rows["primitive_id"] = event_rows["primitive_id"].astype(str)
    rows: list[dict[str, Any]] = []
    for case in case_rows.itertuples(index=False):
        for role_side, primitive_id in [
            ("candidate", case.candidate_primitive_id),
            ("control", case.control_primitive_id),
        ]:
            selected = event_rows[event_rows["primitive_id"].eq(str(primitive_id))]
            for event in selected.itertuples(index=False):
                event_dict = event._asdict()
                host_context = str(event_dict.get("host_context_class", ""))
                split_class = str(event_dict.get("split_vector_class", ""))
                rows.append(
                    {
                        "panel_case_id": case.panel_case_id,
                        "panel_case_rank": case.panel_case_rank,
                        "panel_scope": case.panel_scope,
                        "analysis_tier": case.analysis_tier,
                        "strict_core_v0": case.strict_core_v0,
                        "role_side": role_side,
                        "primitive_id": primitive_id,
                        "event_id": str(event_dict.get("event_id", "")),
                        "branch": str(event_dict.get("branch", "")),
                        "comparison_seed": event_dict.get("comparison_seed", ""),
                        "event_pre_endpoint_role": _event_pre_endpoint_role(
                            role_side, host_context, split_class
                        ),
                        "split_vector_class": split_class,
                        "host_context_class": host_context,
                        "dominant_host_handle_id": str(
                            event_dict.get("dominant_host_handle_id", "")
                        ),
                        "dominant_host_is_source_ref": event_dict.get(
                            "dominant_host_is_source_ref", ""
                        ),
                        "top1_endpoint_handle_id": str(
                            event_dict.get("top1_endpoint_handle_id", "")
                        ),
                        "top1_segment_share_ref_weight": event_dict.get(
                            "top1_segment_share_ref_weight", ""
                        ),
                        "top2_segment_share_ref_weight": event_dict.get(
                            "top2_segment_share_ref_weight", ""
                        ),
                        "effective_segment_count": event_dict.get(
                            "effective_segment_count", ""
                        ),
                        "fragmentation_index": event_dict.get("fragmentation_index", ""),
                        "split_segment_count_ge5_weight": event_dict.get(
                            "split_segment_count_ge5_weight", ""
                        ),
                        "is_strong_boundary_seed": event_dict.get(
                            "is_strong_boundary_seed", ""
                        ),
                        "is_severe_boundary_seed": event_dict.get(
                            "is_severe_boundary_seed", ""
                        ),
                        "endpoint_success_unit": case.endpoint_success_unit,
                        "candidate_caveat_flags": case.candidate_caveat_flags,
                    }
                )
    return _with_claim_columns(pd.DataFrame(rows))


def _signature_complexity(role_side: str, group: pd.DataFrame) -> str:
    if role_side == "control":
        return "source_preserved_control_signature"
    source_preserved = int(
        group["host_context_class"].astype(str).eq("source_host_preserved").sum()
    )
    dominant_hosts = int(group["dominant_host_handle_id"].nunique())
    top1_handles = int(group["top1_endpoint_handle_id"].nunique())
    if source_preserved > 0:
        return "mixed_host_context"
    if dominant_hosts >= 3 or top1_handles >= 6:
        return "broad_multi_target"
    return "simple_external_target"


def _endpoint_signature_rows(case_rows: pd.DataFrame, event_role_rows: pd.DataFrame) -> pd.DataFrame:
    case_lookup = case_rows.set_index("panel_case_id")
    rows: list[dict[str, Any]] = []
    for (case_id, role_side), group in event_role_rows.groupby(
        ["panel_case_id", "role_side"], sort=True
    ):
        case = case_lookup.loc[str(case_id)]
        top1_counts = group["top1_endpoint_handle_id"].astype(str).value_counts()
        top1_reuse_max = int(top1_counts.max()) if not top1_counts.empty else 0
        top1_unique = int(group["top1_endpoint_handle_id"].nunique())
        dominant_host_unique = int(group["dominant_host_handle_id"].nunique())
        rows.append(
            {
                "endpoint_signature_id": f"{case_id}__{role_side}_event_family",
                "panel_case_id": case_id,
                "panel_case_rank": int(case["panel_case_rank"]),
                "panel_scope": str(case["panel_scope"]),
                "analysis_tier": str(case["analysis_tier"]),
                "strict_core_v0": bool(case["strict_core_v0"]),
                "role_side": role_side,
                "primitive_id": str(
                    case["candidate_primitive_id"]
                    if role_side == "candidate"
                    else case["control_primitive_id"]
                ),
                "endpoint_success_unit": "endpoint_family_signature_distance",
                "signature_target_complexity": _signature_complexity(role_side, group),
                "event_count": int(len(group)),
                "comparison_seed_count": int(group["comparison_seed"].nunique()),
                "dominant_host_unique_count": dominant_host_unique,
                "top1_endpoint_unique_count": top1_unique,
                "top1_endpoint_reuse_max_within_signature": top1_reuse_max,
                "external_host_event_count": int(
                    group["host_context_class"].astype(str).eq("external_host_absorption").sum()
                ),
                "source_preserved_event_count": int(
                    group["host_context_class"].astype(str).eq("source_host_preserved").sum()
                ),
                "event_pre_endpoint_role_counts": _join_counts(
                    group["event_pre_endpoint_role"]
                ),
                "split_vector_class_counts": _join_counts(group["split_vector_class"]),
                "host_context_class_counts": _join_counts(group["host_context_class"]),
                "dominant_host_handle_ids": _join_unique(
                    group["dominant_host_handle_id"], max_items=24
                ),
                "top1_endpoint_handle_ids": _join_unique(
                    group["top1_endpoint_handle_id"], max_items=24
                ),
                "median_top1_segment_share_ref_weight": float(
                    pd.to_numeric(
                        group["top1_segment_share_ref_weight"], errors="coerce"
                    ).median()
                ),
                "median_effective_segment_count": float(
                    pd.to_numeric(group["effective_segment_count"], errors="coerce").median()
                ),
                "endpoint_signature_readout_contract": (
                    "future replay must score distance to this event-family "
                    "signature; single top1 endpoint handles are diagnostic "
                    "members, not standalone hits"
                ),
                "candidate_caveat_flags": str(case["candidate_caveat_flags"]),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows))


def _control_pool(primitive_rows: pd.DataFrame) -> pd.DataFrame:
    rows = primitive_rows.copy()
    return rows[
        rows["analog_tier"].astype(str).eq("source_preserved_multifragment_control_like")
        | (
            rows["endpoint_host_scope"].astype(str).eq("all_events_source_host_preserved")
            & ~rows["is_joint_weak_pair_analog_candidate"].astype(bool)
        )
    ].copy()


def _control_distance(candidate: pd.Series, control: pd.Series) -> float:
    return float(
        abs(float(control["event_count"]) - float(candidate["candidate_event_count"]))
        + abs(
            float(control["effective_segment_count_median"])
            - float(candidate["candidate_effective_segment_count_median"])
        )
        + abs(
            float(control["top1_segment_share_ref_weight_median"])
            - float(candidate["candidate_top1_segment_share_ref_weight_median"])
        )
    )


def _linear_assignment(cost_matrix: list[list[float]]) -> tuple[list[int], list[int], str]:
    try:
        from scipy.optimize import linear_sum_assignment

        row_indexes, column_indexes = linear_sum_assignment(cost_matrix)
        return list(row_indexes), list(column_indexes), "scipy_linear_sum_assignment"
    except Exception:
        remaining_rows = set(range(len(cost_matrix)))
        remaining_columns = set(range(len(cost_matrix[0]))) if cost_matrix else set()
        pairs: list[tuple[int, int]] = []
        while remaining_rows and remaining_columns:
            best: tuple[float, int, int] | None = None
            for row in remaining_rows:
                for column in remaining_columns:
                    candidate = (float(cost_matrix[row][column]), row, column)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                break
            _, row, column = best
            pairs.append((row, column))
            remaining_rows.remove(row)
            remaining_columns.remove(column)
        return (
            [row for row, _ in pairs],
            [column for _, column in pairs],
            "greedy_fallback",
        )


def _control_sensitivity_rows(
    case_rows: pd.DataFrame,
    primitive_rows: pd.DataFrame,
) -> pd.DataFrame:
    control_pool = _control_pool(primitive_rows)
    core_cases = case_rows[case_rows["panel_scope"].astype(str).eq("core_tier1_panel")].copy()
    rows: list[dict[str, Any]] = []
    for branch, branch_cases in core_cases.groupby("candidate_branch", sort=True):
        branch_controls = control_pool[control_pool["branch"].astype(str).eq(str(branch))]
        if branch_controls.empty:
            branch_controls = control_pool
        branch_cases = branch_cases.sort_values("panel_case_rank")
        branch_controls = branch_controls.sort_values(["branch", "primitive_id"])
        cost_matrix: list[list[float]] = [
            [
                _control_distance(case, control)
                for _, control in branch_controls.iterrows()
            ]
            for _, case in branch_cases.iterrows()
        ]
        row_indexes, control_indexes, assignment_method = _linear_assignment(cost_matrix)
        for row_index, control_index in zip(row_indexes, control_indexes):
            case = branch_cases.iloc[row_index]
            assigned = branch_controls.iloc[control_index]
            unique_distance = float(cost_matrix[row_index][control_index])
            rows.append(
                {
                    "sensitivity_scope": "core_tier1_one_to_one_same_branch",
                    "assignment_method": assignment_method,
                    "panel_case_id": str(case["panel_case_id"]),
                    "panel_case_rank": int(case["panel_case_rank"]),
                    "analysis_tier": str(case["analysis_tier"]),
                    "strict_core_v0": bool(case["strict_core_v0"]),
                    "candidate_branch": str(case["candidate_branch"]),
                    "candidate_primitive_id": str(case["candidate_primitive_id"]),
                    "reusable_control_primitive_id": str(case["control_primitive_id"]),
                    "reusable_control_reuse_count": int(case["reusable_control_reuse_count"]),
                    "reusable_matching_distance": float(case["matching_distance"]),
                    "one_to_one_control_primitive_id": str(assigned["primitive_id"]),
                    "one_to_one_control_source_family_id": str(assigned["source_family_id"]),
                    "one_to_one_control_analog_tier": str(assigned["analog_tier"]),
                    "one_to_one_control_event_count": int(assigned["event_count"]),
                    "one_to_one_control_effective_segment_count_median": float(
                        assigned["effective_segment_count_median"]
                    ),
                    "one_to_one_control_top1_segment_share_ref_weight_median": float(
                        assigned["top1_segment_share_ref_weight_median"]
                    ),
                    "one_to_one_matching_distance": unique_distance,
                    "distance_delta_vs_reusable": unique_distance
                    - float(case["matching_distance"]),
                    "sensitivity_read": (
                        "core-only same-branch one-to-one control sensitivity; "
                        "use as robustness check, not as the main reusable-control panel"
                    ),
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["panel_case_rank"]))


def _endpoint_replay_contract() -> pd.DataFrame:
    rows = [
        {
            "contract_id": "R1_input_freeze",
            "contract_question": "Are panel cases, analysis tiers, roles, controls, graph scope, gamma, seed budget, and replay code hash frozen before replay?",
            "required_future_artifact": "replay_config.json",
            "required_fields": "panel_case_id,analysis_tier,role_id,graph_scope,gamma,seed_list,budget,code_hash,input_hash",
            "pass_condition": "all future replay rows can be traced to frozen panel case, analysis tier, and role IDs",
            "blocked_claim_if_missing": "method success",
        },
        {
            "contract_id": "R2_role_isolation",
            "contract_question": "Are candidate and control roles executed symmetrically?",
            "required_future_artifact": "endpoint_replay_attempt_rows.csv",
            "required_fields": "panel_case_id,role_side,method_seed,attempt_id,terminal_endpoint_signature",
            "pass_condition": "candidate and matched control use identical branch, gamma, seed budget, and attempt budget",
            "blocked_claim_if_missing": "candidate/control contrast",
        },
        {
            "contract_id": "R3_endpoint_signature",
            "contract_question": "Is endpoint identity read against endpoint-family signatures, not a single top1 endpoint handle?",
            "required_future_artifact": "endpoint_signature_rows.csv",
            "required_fields": "endpoint_signature_id,target_signature_unit,terminal_endpoint_signature,distance_to_candidate_event_family,distance_to_control_event_family,nearest_known_endpoint",
            "pass_condition": "endpoint hit or miss is computed from predeclared event-family signature fields; single endpoint handles are diagnostic members only",
            "blocked_claim_if_missing": "basin-discovery readout",
        },
        {
            "contract_id": "R4_candidate_advantage",
            "contract_question": "Does the candidate role find endpoint-family signatures more often or earlier than the matched reusable control?",
            "required_future_artifact": "endpoint_replay_case_summary.csv",
            "required_fields": "analysis_tier,candidate_signature_hit_rate,control_signature_hit_rate,candidate_first_hit_attempt,control_first_hit_attempt",
            "pass_condition": "primary readout is computed on strict_core_v0 first, then full_core and reserve as sensitivity tiers",
            "blocked_claim_if_missing": "real-data method signal",
        },
        {
            "contract_id": "R5_negative_controls",
            "contract_question": "Do non-analog or source-preserved controls remain claim-disabled unless they independently pass endpoint readout?",
            "required_future_artifact": "endpoint_replay_control_rows.csv",
            "required_fields": "control_primitive_id,control_pre_endpoint_role,control_hit_rate,control_positive_attempt_count",
            "pass_condition": "controls are reported as controls, not reclassified post hoc as wins",
            "blocked_claim_if_missing": "mechanism specificity",
        },
        {
            "contract_id": "R6_control_sensitivity",
            "contract_question": "Does the replay report the reusable-control main panel and core-only one-to-one control sensitivity separately?",
            "required_future_artifact": "endpoint_replay_control_sensitivity_summary.csv",
            "required_fields": "panel_case_id,reusable_control_primitive_id,one_to_one_control_primitive_id,reusable_hit_rate,one_to_one_hit_rate",
            "pass_condition": "control-anchor reuse is not hidden; one-to-one sensitivity is reported as robustness evidence",
            "blocked_claim_if_missing": "control robustness",
        },
        {
            "contract_id": "R7_claim_boundary",
            "contract_question": "Are wall/pathway, quality/cost, and algorithm claims kept closed after endpoint replay unless separate gates exist?",
            "required_future_artifact": "claim_boundary_gate_matrix.csv",
            "required_fields": "route_execution_status,wall_promotion_status,quality_cost_status,algorithm_claim_status",
            "pass_condition": "endpoint replay can only support local method signal, not wall/pathway or quality claims",
            "blocked_claim_if_missing": "claim-boundary integrity",
        },
    ]
    return _with_claim_columns(pd.DataFrame(rows))


def _gate_matrix(
    case_rows: pd.DataFrame,
    role_rows: pd.DataFrame,
    event_role_rows: pd.DataFrame,
    endpoint_signature_rows: pd.DataFrame,
    control_sensitivity_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
) -> pd.DataFrame:
    core_cases = case_rows[case_rows["panel_scope"].astype(str).eq("core_tier1_panel")]
    reserve_cases = case_rows[
        case_rows["panel_scope"].astype(str).eq("reserve_lower_tier_panel")
    ]
    strict_cases = case_rows[case_rows["strict_core_v0"].astype(bool)]
    full_core_caveated = core_cases[~core_cases["strict_core_v0"].astype(bool)]
    branch_count = int(core_cases["candidate_branch"].nunique()) if not core_cases.empty else 0
    case_ids = set(case_rows["panel_case_id"].astype(str))
    event_case_ids = set(event_role_rows["panel_case_id"].astype(str))
    missing_event_cases = sorted(case_ids - event_case_ids)
    reusable_unique_controls = int(case_rows["control_primitive_id"].nunique())
    reusable_control_reuse_max = int(case_rows["reusable_control_reuse_count"].max())
    core_sensitivity_unique_controls = int(
        control_sensitivity_rows["one_to_one_control_primitive_id"].nunique()
    ) if not control_sensitivity_rows.empty else 0
    rows = [
        {
            "gate_id": "P1_analog_screen_loaded",
            "gate_question": "Did the prior analog screen provide matched candidate/control rows?",
            "status": "pass" if len(case_rows) > 0 else "blocked_no_panel_cases",
            "evidence": f"panel_cases={len(case_rows)}",
            "decision": "local panel design is meaningful only if candidate/control rows exist",
        },
        {
            "gate_id": "P2_analysis_tiers_frozen",
            "gate_question": "Are strict, full-core caveated, and reserve analysis tiers explicit?",
            "status": "pass" if len(strict_cases) > 0 else "blocked_no_strict_core_cases",
            "evidence": (
                f"strict_core_v0={len(strict_cases)}, "
                f"full_core_caveated={len(full_core_caveated)}, "
                f"reserve={len(reserve_cases)}"
            ),
            "decision": "endpoint replay must use strict_core_v0 as primary denominator",
        },
        {
            "gate_id": "P3_reusable_controls_attached_with_reuse_caveat",
            "gate_question": "Does every panel case have a matched reusable control, and is control-anchor reuse visible?",
            "status": (
                "caveat_reused_control_anchors"
                if reusable_unique_controls < len(case_rows)
                else "pass"
            ),
            "evidence": (
                f"matched_control_cases={case_rows['control_primitive_id'].astype(str).ne('').sum()}, "
                f"unique_reusable_controls={reusable_unique_controls}, "
                f"max_reuse={reusable_control_reuse_max}"
            ),
            "decision": "main panel uses reusable nearest controls; do not count it as 30 independent control anchors",
        },
        {
            "gate_id": "P4_core_one_to_one_sensitivity_written",
            "gate_question": "Is a core-only one-to-one control sensitivity panel written?",
            "status": (
                "pass"
                if len(control_sensitivity_rows) == len(core_cases)
                and core_sensitivity_unique_controls == len(core_cases)
                else "blocked_missing_core_control_sensitivity"
            ),
            "evidence": (
                f"core_sensitivity_rows={len(control_sensitivity_rows)}, "
                f"unique_one_to_one_controls={core_sensitivity_unique_controls}"
            ),
            "decision": "control robustness should be evaluated separately from the reusable-control main panel",
        },
        {
            "gate_id": "P5_event_role_rows_available",
            "gate_question": "Can selected candidate/control primitives be traced to event-level role rows?",
            "status": "pass" if not missing_event_cases else "blocked_missing_event_roles",
            "evidence": f"event_role_rows={len(event_role_rows)}, missing_event_case_count={len(missing_event_cases)}",
            "decision": "event roles are required before replay design can be audited",
        },
        {
            "gate_id": "P6_endpoint_family_signatures_written",
            "gate_question": "Are endpoint-family signature rows written before replay?",
            "status": (
                "pass"
                if len(endpoint_signature_rows)
                == int(case_rows["panel_case_id"].nunique()) * 2
                else "blocked_missing_endpoint_signature_rows"
            ),
            "evidence": f"endpoint_signature_rows={len(endpoint_signature_rows)}",
            "decision": "future replay must use signature distance, not single endpoint-handle hits",
        },
        {
            "gate_id": "P7_core_branch_balance_visible",
            "gate_question": "Is branch composition visible for the core panel?",
            "status": "pass" if branch_count >= 2 else "caveat_single_branch_core_panel",
            "evidence": ";".join(
                f"{k}:{v}"
                for k, v in core_cases["candidate_branch"].value_counts().sort_index().items()
            ),
            "decision": "branch balance is descriptive, not a success gate",
        },
        {
            "gate_id": "P8_endpoint_replay_contract_written",
            "gate_question": "Is a future endpoint-replay readout contract written before execution?",
            "status": "pass" if len(contract_rows) >= 7 else "blocked_missing_contract",
            "evidence": f"contract_rows={len(contract_rows)}",
            "decision": "replay should not start without the contract",
        },
        {
            "gate_id": "P9_execution_boundary_closed",
            "gate_question": "Does this panel stop before endpoint replay, route/wall, quality/cost, and algorithm claims?",
            "status": "closed_excluded_by_design",
            "evidence": CLAIM_BOUNDARY,
            "decision": "panel design can justify replay setup only, not method success",
        },
    ]
    return _with_claim_columns(pd.DataFrame(rows))


def _readiness(gates: pd.DataFrame) -> str:
    statuses = set(gates["status"].astype(str))
    if any(status.startswith("blocked") for status in statuses):
        return "blocked_local_panel_design_incomplete"
    if any(status.startswith("caveat") for status in statuses):
        return "caveated_ready_for_replay_design_review"
    return "ready_for_endpoint_replay_design_review"


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.loc[:, columns].head(max_rows).iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        table.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        table.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(table)


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    case_rows: pd.DataFrame,
    role_rows: pd.DataFrame,
    event_role_rows: pd.DataFrame,
    endpoint_signature_rows: pd.DataFrame,
    control_sensitivity_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    core_cases = case_rows[case_rows["panel_scope"].astype(str).eq("core_tier1_panel")]
    strict_cases = case_rows[case_rows["strict_core_v0"].astype(bool)]
    full_core_caveated = core_cases[~core_cases["strict_core_v0"].astype(bool)]
    lines = [
        "# NanoClustering Joint Weak-Pair Local Panel Design",
        "",
        f"- analog_screen_dir: `{summary['analog_screen_dir']}`",
        f"- measurement_dir: `{summary['measurement_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- readiness: `{summary['readiness']}`",
        f"- panel_case_count: `{summary['panel_case_count']}`",
        f"- core_tier1_case_count: `{summary['core_tier1_case_count']}`",
        f"- strict_core_v0_count: `{summary['strict_core_v0_count']}`",
        f"- full_core_caveated_count: `{summary['full_core_caveated_count']}`",
        f"- reserve_case_count: `{summary['reserve_case_count']}`",
        f"- reusable_control_unique_count: `{summary['reusable_control_unique_count']}`",
        f"- core_one_to_one_control_unique_count: `{summary['core_one_to_one_control_unique_count']}`",
        f"- role_row_count: `{summary['role_row_count']}`",
        f"- event_role_row_count: `{summary['event_role_row_count']}`",
        f"- endpoint_signature_row_count: `{summary['endpoint_signature_row_count']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(gates, ["gate_id", "status", "evidence", "decision"], max_rows=12),
        "",
        "## Analysis Tiers",
        "",
        "- `strict_core_v0` is the primary future replay denominator.",
        "- `full_core_caveated_sensitivity` keeps tier-1 cases with mixed host context, residual definition debt, or weaker reusable-control matching.",
        "- `reserve_exploratory` keeps lower-tier recurrent cases outside the primary claim path.",
        "",
        "## Strict Core v0 Cases",
        "",
        _markdown_table(
            strict_cases,
            [
                "panel_case_id",
                "candidate_primitive_id",
                "candidate_branch",
                "candidate_analog_score",
                "candidate_event_count",
                "candidate_split_vector_class_mode",
                "control_primitive_id",
                "matching_distance",
            ],
            max_rows=20,
        ),
        "",
        "## Full-Core Caveated Cases",
        "",
        _markdown_table(
            full_core_caveated,
            [
                "panel_case_id",
                "candidate_primitive_id",
                "candidate_branch",
                "candidate_caveat_flags",
                "control_primitive_id",
                "matching_distance",
            ],
            max_rows=20,
        ),
        "",
        "## Core Tier-1 Panel Cases",
        "",
        _markdown_table(
            core_cases,
            [
                "panel_case_id",
                "analysis_tier",
                "candidate_primitive_id",
                "candidate_branch",
                "candidate_analog_score",
                "candidate_event_count",
                "candidate_split_vector_class_mode",
                "candidate_host_context_class_mode",
                "candidate_caveat_flags",
                "control_primitive_id",
                "reusable_control_reuse_count",
                "matching_distance",
            ],
            max_rows=20,
        ),
        "",
        "## Core One-To-One Control Sensitivity",
        "",
        _markdown_table(
            control_sensitivity_rows,
            [
                "panel_case_id",
                "candidate_primitive_id",
                "reusable_control_primitive_id",
                "one_to_one_control_primitive_id",
                "reusable_matching_distance",
                "one_to_one_matching_distance",
                "distance_delta_vs_reusable",
            ],
            max_rows=20,
        ),
        "",
        "## Frozen Role Rows",
        "",
        _markdown_table(
            role_rows,
            [
                "panel_case_id",
                "analysis_tier",
                "role_side",
                "primitive_id",
                "pre_endpoint_role",
                "event_count",
                "split_vector_class_mode",
                "host_context_class_mode",
            ],
            max_rows=20,
        ),
        "",
        "## Endpoint-Family Signatures",
        "",
        _markdown_table(
            endpoint_signature_rows,
            [
                "endpoint_signature_id",
                "role_side",
                "analysis_tier",
                "signature_target_complexity",
                "event_count",
                "dominant_host_unique_count",
                "top1_endpoint_unique_count",
                "top1_endpoint_reuse_max_within_signature",
            ],
            max_rows=20,
        ),
        "",
        "## Endpoint Replay Contract",
        "",
        _markdown_table(
            contract_rows,
            [
                "contract_id",
                "required_future_artifact",
                "required_fields",
                "blocked_claim_if_missing",
            ],
            max_rows=10,
        ),
        "",
        "## Event Role Coverage",
        "",
        f"- event_role_rows: `{len(event_role_rows)}`",
        f"- candidate_event_role_rows: `{int((event_role_rows['role_side'].astype(str) == 'candidate').sum())}`",
        f"- control_event_role_rows: `{int((event_role_rows['role_side'].astype(str) == 'control').sum())}`",
        f"- candidate_endpoint_signature_rows: `{int((endpoint_signature_rows['role_side'].astype(str) == 'candidate').sum())}`",
        f"- control_endpoint_signature_rows: `{int((endpoint_signature_rows['role_side'].astype(str) == 'control').sum())}`",
        "",
        "## Interpretation Boundary",
        "",
        "- This is a local panel design artifact, not a replay result.",
        "- The valid next use is endpoint-replay implementation against the frozen `strict_core_v0` case and role IDs first.",
        "- Endpoint success must be a family/signature-distance readout; single top1 endpoint handles are diagnostic members only.",
        "- Reusable-control results and core one-to-one-control sensitivity must be reported separately.",
        "- Route/pathway, wall, quality/cost, and algorithm claims remain closed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def design_panel(
    *,
    analog_screen_dir: Path,
    measurement_dir: Path,
    output_dir: Path,
    max_cases: int | None,
    force: bool,
) -> dict[str, Any]:
    if output_dir.exists() and not force:
        raise FileExistsError(f"{output_dir} exists; pass --force to overwrite outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    primitive_rows = _read_csv(analog_screen_dir / ANALOG_PRIMITIVE_ROWS_CSV)
    matched_controls = _read_csv(analog_screen_dir / ANALOG_CONTROL_ROWS_CSV)
    event_rows = _read_csv(measurement_dir / MEASUREMENT_EVENT_ROWS_CSV)

    case_rows = _build_case_rows(
        primitive_rows=primitive_rows,
        matched_controls=matched_controls,
        max_cases=max_cases,
    )
    role_rows = _role_rows(case_rows)
    event_role_rows = _event_role_rows(case_rows, event_rows)
    endpoint_signature_rows = _endpoint_signature_rows(case_rows, event_role_rows)
    control_sensitivity_rows = _control_sensitivity_rows(case_rows, primitive_rows)
    contract_rows = _endpoint_replay_contract()
    gates = _gate_matrix(
        case_rows,
        role_rows,
        event_role_rows,
        endpoint_signature_rows,
        control_sensitivity_rows,
        contract_rows,
    )
    readiness = _readiness(gates)

    _write_csv(case_rows, output_dir / PANEL_CASE_ROWS_CSV)
    _write_csv(role_rows, output_dir / PANEL_ROLE_ROWS_CSV)
    _write_csv(event_role_rows, output_dir / PANEL_EVENT_ROLE_ROWS_CSV)
    _write_csv(endpoint_signature_rows, output_dir / PANEL_ENDPOINT_SIGNATURE_ROWS_CSV)
    _write_csv(control_sensitivity_rows, output_dir / CONTROL_SENSITIVITY_ROWS_CSV)
    _write_csv(contract_rows, output_dir / ENDPOINT_REPLAY_CONTRACT_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)

    config = {
        "analog_screen_dir": _rel(analog_screen_dir),
        "measurement_dir": _rel(measurement_dir),
        "output_dir": _rel(output_dir),
        "max_cases": max_cases,
        "claim_boundary": CLAIM_BOUNDARY,
        "role_freeze_status": ROLE_FREEZE_STATUS,
        "replay_execution_status": REPLAY_EXECUTION_STATUS,
        "strict_matching_distance_max": STRICT_MATCHING_DISTANCE_MAX,
        "primary_analysis_tier": "strict_core_v0_primary",
        "control_sensitivity_scope": "core_tier1_one_to_one_same_branch",
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    core_cases = case_rows[case_rows["panel_scope"].astype(str).eq("core_tier1_panel")]
    reserve_cases = case_rows[
        case_rows["panel_scope"].astype(str).eq("reserve_lower_tier_panel")
    ]
    strict_cases = case_rows[case_rows["strict_core_v0"].astype(bool)]
    full_core_caveated = core_cases[~core_cases["strict_core_v0"].astype(bool)]
    summary = {
        "analog_screen_dir": _rel(analog_screen_dir),
        "measurement_dir": _rel(measurement_dir),
        "output_dir": _rel(output_dir),
        "readiness": readiness,
        "panel_case_count": int(len(case_rows)),
        "core_tier1_case_count": int(len(core_cases)),
        "strict_core_v0_count": int(len(strict_cases)),
        "full_core_caveated_count": int(len(full_core_caveated)),
        "reserve_case_count": int(len(reserve_cases)),
        "reusable_control_unique_count": int(case_rows["control_primitive_id"].nunique()),
        "reusable_control_reuse_max": int(case_rows["reusable_control_reuse_count"].max()),
        "core_one_to_one_control_unique_count": int(
            control_sensitivity_rows["one_to_one_control_primitive_id"].nunique()
        ),
        "core_one_to_one_control_mean_distance": float(
            control_sensitivity_rows["one_to_one_matching_distance"].mean()
        ),
        "core_reusable_control_mean_distance": float(core_cases["matching_distance"].mean()),
        "role_row_count": int(len(role_rows)),
        "event_role_row_count": int(len(event_role_rows)),
        "endpoint_signature_row_count": int(len(endpoint_signature_rows)),
        "candidate_event_role_row_count": int(
            (event_role_rows["role_side"].astype(str) == "candidate").sum()
        ),
        "control_event_role_row_count": int(
            (event_role_rows["role_side"].astype(str) == "control").sum()
        ),
        "core_branch_counts": {
            str(key): int(value)
            for key, value in core_cases["candidate_branch"].value_counts().sort_index().to_dict().items()
        },
        "strict_core_branch_counts": {
            str(key): int(value)
            for key, value in strict_cases["candidate_branch"].value_counts().sort_index().to_dict().items()
        },
        "analysis_tier_counts": {
            str(key): int(value)
            for key, value in case_rows["analysis_tier"].value_counts().sort_index().to_dict().items()
        },
        "candidate_caveat_flag_counts": {
            str(key): int(value)
            for key, value in case_rows["candidate_caveat_flags"].value_counts().sort_index().to_dict().items()
        },
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gates["status"].value_counts().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            PANEL_CASE_ROWS_CSV,
            PANEL_ROLE_ROWS_CSV,
            PANEL_EVENT_ROLE_ROWS_CSV,
            PANEL_ENDPOINT_SIGNATURE_ROWS_CSV,
            CONTROL_SENSITIVITY_ROWS_CSV,
            ENDPOINT_REPLAY_CONTRACT_CSV,
            GATE_MATRIX_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / REPORT_MD,
        summary=summary,
        case_rows=case_rows,
        role_rows=role_rows,
        event_role_rows=event_role_rows,
        endpoint_signature_rows=endpoint_signature_rows,
        control_sensitivity_rows=control_sensitivity_rows,
        contract_rows=contract_rows,
        gates=gates,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analog-screen-dir", type=Path, default=DEFAULT_ANALOG_SCREEN_DIR)
    parser.add_argument("--measurement-dir", type=Path, default=DEFAULT_MEASUREMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = design_panel(
        analog_screen_dir=args.analog_screen_dir,
        measurement_dir=args.measurement_dir,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        force=args.force,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
