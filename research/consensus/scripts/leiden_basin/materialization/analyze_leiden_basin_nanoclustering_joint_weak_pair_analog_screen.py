#!/usr/bin/env python3
"""Screen NanoClustering v2.2 primitives for joint weak-pair analogs.

This is a read-only analog screen after the tiny CPM Stress 4 v1.2 result. It
does not run Leiden, create method candidates, execute routes, promote walls, or
measure quality/cost. It asks whether existing NanoClustering endpoint-vector
artifacts contain source-family structures that resemble the tiny-demo joint
weak-pair mechanism: recurrent multi-fragment endpoint alternatives where a
single source family has several simultaneously relevant split handles and
possibly competing host contexts.
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
DEFAULT_MEASUREMENT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_v2_2_measurement_panel_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_joint_weak_pair_analog_screen_20260601"
)

MEASUREMENT_ROWS_CSV = "nanoclustering_v2_2_accepted_primitive_measurement_rows.csv"
EVENT_ROWS_CSV = "nanoclustering_v2_2_accepted_primitive_event_measurement_rows.csv"

PRIMITIVE_ANALOG_ROWS_CSV = "nanoclustering_joint_weak_pair_analog_primitive_rows.csv"
SOURCE_FAMILY_ROWS_CSV = "nanoclustering_joint_weak_pair_analog_source_family_rows.csv"
MATCHED_CONTROL_ROWS_CSV = "nanoclustering_joint_weak_pair_analog_matched_control_rows.csv"
GATE_MATRIX_CSV = "nanoclustering_joint_weak_pair_analog_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_joint_weak_pair_analog_summary.json"
REPORT_MD = "nanoclustering_joint_weak_pair_analog_report.md"
CONFIG_JSON = "nanoclustering_joint_weak_pair_analog_config.json"

CLAIM_BOUNDARY = (
    "NanoClustering joint weak-pair analog screen only; reads v2.2 measurement "
    "and endpoint-vector rows, does not run clustering, construct method "
    "candidates, execute routes/pathways, promote walls, measure quality/cost, "
    "or support algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "not_executed_read_only_analog_screen"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_joint_weak_pair_analog_screen"

MULTI_FRAGMENT_CLASSES = {
    "balanced_multi_handle_split_vector",
    "balanced_two_way_split_vector",
    "diffuse_multiway_fragmentation_vector",
    "multi_handle_fragmentation_vector",
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


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = frame.copy()
    for column in columns:
        if column in rows:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def _mode(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return ""
    counts = clean.value_counts()
    max_count = counts.max()
    return sorted(counts[counts.eq(max_count)].index.tolist())[0]


def _join_counts(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return ""
    counts = clean.value_counts()
    return ";".join(f"{key}:{int(counts[key])}" for key in sorted(counts.index))


def _join_unique(values: pd.Series, *, limit: int = 30) -> str:
    unique = sorted({str(value) for value in values.dropna() if str(value)})
    if len(unique) > limit:
        return ";".join(unique[:limit]) + f";...(+{len(unique) - limit})"
    return ";".join(unique)


def _event_stats(event_rows: pd.DataFrame) -> pd.DataFrame:
    rows = _numeric(
        event_rows,
        [
            "top2_segment_share_ref_weight",
            "split_segment_count_ge5_weight",
            "merge_contributor_count_ge5_weight",
            "effective_segment_count",
            "top1_segment_share_ref_weight",
            "fragmentation_index",
        ],
    )
    grouped: list[dict[str, Any]] = []
    for primitive_id, group in rows.groupby("primitive_id", sort=True):
        grouped.append(
            {
                "primitive_id": str(primitive_id),
                "event_row_count": int(group["event_id"].nunique()),
                "event_dominant_host_distinct_count": int(group["dominant_host_handle_id"].dropna().astype(str).nunique()),
                "event_dominant_host_ids": _join_unique(group["dominant_host_handle_id"]),
                "event_split_vector_class_counts": _join_counts(group["split_vector_class"]),
                "event_host_context_class_counts": _join_counts(group["host_context_class"]),
                "top2_segment_share_ref_weight_median": float(group["top2_segment_share_ref_weight"].median()),
                "split_segment_count_ge5_weight_median_from_events": float(
                    group["split_segment_count_ge5_weight"].median()
                ),
                "merge_contributor_count_ge5_weight_median_from_events": float(
                    group["merge_contributor_count_ge5_weight"].median()
                ),
                "max_event_effective_segment_count": float(group["effective_segment_count"].max()),
                "min_event_top1_segment_share": float(group["top1_segment_share_ref_weight"].min()),
                "max_event_fragmentation_index": float(group["fragmentation_index"].max()),
            }
        )
    return pd.DataFrame(grouped)


def _source_context(measurement_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_family_id, group in measurement_rows.groupby("source_family_id", sort=True):
        rows.append(
            {
                "source_family_id": str(source_family_id),
                "source_family_accepted_primitive_count_observed": int(group["primitive_id"].nunique()),
                "source_family_event_count_observed": int(group["event_count"].sum()),
                "source_family_distinct_host_mode_count": int(
                    group["dominant_host_handle_id_mode"].dropna().astype(str).nunique()
                ),
                "source_family_dominant_host_modes": _join_unique(group["dominant_host_handle_id_mode"]),
                "source_family_split_mode_counts": _join_counts(group["split_vector_class_mode"]),
                "source_family_host_mode_counts": _join_counts(group["host_context_class_mode"]),
                "source_family_max_effective_segment_count_median": float(
                    group["effective_segment_count_median"].max()
                ),
                "source_family_min_top1_share_median": float(
                    group["top1_segment_share_ref_weight_median"].min()
                ),
                "source_family_max_fragmentation_index_median": float(
                    group["fragmentation_index_median"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _analog_score(row: pd.Series) -> int:
    score = 0
    event_count = int(row["event_count"])
    effective = float(row["effective_segment_count_median"])
    top1 = float(row["top1_segment_share_ref_weight_median"])
    split_count = float(row["split_segment_count_ge5_weight_median"])
    top2 = float(row["top2_segment_share_ref_weight_median"])
    host_mode_share = float(row["dominant_host_handle_id_mode_share"])
    source_host_count = int(row["source_family_distinct_host_mode_count"])
    if event_count >= 5:
        score += 2
    elif event_count >= 3:
        score += 1
    if effective >= 4:
        score += 2
    elif effective >= 3:
        score += 1
    if top1 < 0.35:
        score += 2
    elif top1 < 0.5:
        score += 1
    if split_count >= 4:
        score += 2
    elif split_count >= 3:
        score += 1
    if top2 >= 0.25:
        score += 1
    if row["split_vector_class_mode"] in MULTI_FRAGMENT_CLASSES:
        score += 1
    if row["host_context_class_mode"] == "external_host_absorption":
        score += 1
    if host_mode_share < 0.9 or source_host_count >= 2:
        score += 1
    if int(row["source_family_accepted_primitive_count"]) >= 2:
        score += 1
    return int(score)


def _analog_tier(row: pd.Series) -> str:
    recurrent = int(row["event_count"]) >= 5
    moderate = int(row["event_count"]) >= 3
    effective = float(row["effective_segment_count_median"])
    top1 = float(row["top1_segment_share_ref_weight_median"])
    split_count = float(row["split_segment_count_ge5_weight_median"])
    host_mode_share = float(row["dominant_host_handle_id_mode_share"])
    source_host_count = int(row["source_family_distinct_host_mode_count"])
    external = row["host_context_class_mode"] == "external_host_absorption"
    multi_fragment = row["split_vector_class_mode"] in MULTI_FRAGMENT_CLASSES
    host_competition = host_mode_share < 0.9 or source_host_count >= 2
    strong_fragment = effective >= 3.0 and top1 < 0.5 and split_count >= 3 and multi_fragment

    if recurrent and strong_fragment and external and host_competition:
        return "tier1_external_multi_fragment_host_competition_analog"
    if recurrent and strong_fragment:
        return "tier2_recurrent_multi_fragment_analog"
    if moderate and strong_fragment:
        return "tier3_moderate_multi_fragment_analog"
    if external and moderate and multi_fragment:
        return "external_multifragment_observer"
    if row["endpoint_host_scope"] == "all_events_source_host_preserved" and strong_fragment:
        return "source_preserved_multifragment_control_like"
    return "not_joint_weak_pair_analog"


def _primitive_rows(measurement_rows: pd.DataFrame, event_rows: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "event_count",
        "source_family_accepted_primitive_count",
        "effective_segment_count_median",
        "top1_segment_share_ref_weight_median",
        "fragmentation_index_median",
        "split_segment_count_ge5_weight_median",
        "merge_contributor_count_ge5_weight_median",
        "dominant_host_handle_id_mode_share",
    ]
    measurement = _numeric(measurement_rows, numeric_columns)
    stats = _event_stats(event_rows)
    source = _source_context(measurement)
    rows = measurement.merge(stats, on="primitive_id", how="left", validate="one_to_one")
    rows = rows.merge(source, on="source_family_id", how="left", validate="many_to_one")
    rows["top2_segment_share_ref_weight_median"] = rows["top2_segment_share_ref_weight_median"].fillna(0.0)
    rows["analog_score"] = rows.apply(_analog_score, axis=1)
    rows["analog_tier"] = rows.apply(_analog_tier, axis=1)
    rows["is_joint_weak_pair_analog_candidate"] = rows["analog_tier"].astype(str).str.startswith("tier")
    rows["screen_read"] = rows["analog_tier"].map(
        {
            "tier1_external_multi_fragment_host_competition_analog": (
                "strongest screen candidate: recurrent multi-fragment endpoint family with external host context "
                "and host competition or multiple source-family host modes"
            ),
            "tier2_recurrent_multi_fragment_analog": (
                "recurrent multi-fragment endpoint family; host context is less competitive or source-preserved"
            ),
            "tier3_moderate_multi_fragment_analog": (
                "moderate-support multi-fragment endpoint family; useful as analog candidate but support is thinner"
            ),
            "external_multifragment_observer": (
                "external-host multi-fragment observer row below the main joint-analog threshold"
            ),
            "source_preserved_multifragment_control_like": (
                "source-preserved multi-fragment control-like contrast row"
            ),
            "not_joint_weak_pair_analog": "outside current joint weak-pair analog screen",
        }
    )
    preferred = [
        "primitive_id",
        "source_family_id",
        "branch",
        "ref_cluster_id",
        "primitive_type",
        "definition_core_v2_2_rule_status",
        "definition_confidence_tier",
        "measurement_support_class",
        "analog_tier",
        "analog_score",
        "is_joint_weak_pair_analog_candidate",
        "screen_read",
        "event_count",
        "source_family_accepted_primitive_count",
        "source_family_distinct_host_mode_count",
        "source_family_dominant_host_modes",
        "split_vector_class_mode",
        "host_context_class_mode",
        "endpoint_host_scope",
        "dominant_host_handle_id_mode",
        "dominant_host_handle_id_mode_share",
        "event_dominant_host_distinct_count",
        "event_dominant_host_ids",
        "top1_endpoint_handle_id_distinct_count",
        "top1_segment_share_ref_weight_median",
        "top2_segment_share_ref_weight_median",
        "effective_segment_count_median",
        "split_segment_count_ge5_weight_median",
        "merge_contributor_count_ge5_weight_median",
        "fragmentation_index_median",
        "target_share_of_best_run_cluster_weight_median",
        "residual_caveat_status",
        "source_family_residual_event_count",
        "event_split_vector_class_counts",
        "event_host_context_class_counts",
        "claim_boundary",
    ]
    existing = [column for column in preferred if column in rows.columns]
    return _with_claim_columns(rows[existing].sort_values(["analog_tier", "analog_score", "event_count"], ascending=[True, False, False]))


def _source_family_rows(primitive_rows: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: list[dict[str, Any]] = []
    candidate_tiers = {
        "tier1_external_multi_fragment_host_competition_analog",
        "tier2_recurrent_multi_fragment_analog",
        "tier3_moderate_multi_fragment_analog",
    }
    for source_family_id, group in primitive_rows.groupby("source_family_id", sort=True):
        candidates = group[group["analog_tier"].isin(candidate_tiers)]
        tier1 = group[group["analog_tier"].eq("tier1_external_multi_fragment_host_competition_analog")]
        if len(tier1):
            status = "source_family_has_tier1_joint_analog_candidate"
        elif len(candidates):
            status = "source_family_has_lower_tier_joint_analog_candidate"
        else:
            status = "source_family_has_no_joint_analog_candidate"
        grouped_rows.append(
            {
                "source_family_id": str(source_family_id),
                "branch": _mode(group["branch"]),
                "primitive_count": int(group["primitive_id"].nunique()),
                "accepted_event_count": int(pd.to_numeric(group["event_count"], errors="coerce").sum()),
                "candidate_primitive_count": int(len(candidates)),
                "tier1_candidate_count": int(len(tier1)),
                "max_analog_score": int(group["analog_score"].max()),
                "source_screen_status": status,
                "analog_tier_counts": _join_counts(group["analog_tier"]),
                "dominant_host_mode_count": int(group["dominant_host_handle_id_mode"].dropna().astype(str).nunique()),
                "dominant_host_modes": _join_unique(group["dominant_host_handle_id_mode"]),
                "split_vector_mode_counts": _join_counts(group["split_vector_class_mode"]),
                "host_context_mode_counts": _join_counts(group["host_context_class_mode"]),
                "max_effective_segment_count_median": float(group["effective_segment_count_median"].max()),
                "min_top1_segment_share_ref_weight_median": float(group["top1_segment_share_ref_weight_median"].min()),
                "max_fragmentation_index_median": float(group["fragmentation_index_median"].max()),
                "residual_caveat_status_counts": _join_counts(group["residual_caveat_status"]),
                "candidate_primitive_ids": _join_unique(candidates["primitive_id"]),
            }
        )
    return _with_claim_columns(pd.DataFrame(grouped_rows).sort_values(["tier1_candidate_count", "candidate_primitive_count", "max_analog_score"], ascending=[False, False, False]))


def _match_controls(primitive_rows: pd.DataFrame, *, max_candidates: int) -> pd.DataFrame:
    candidates = primitive_rows[primitive_rows["is_joint_weak_pair_analog_candidate"].astype(bool)].copy()
    candidates = candidates.sort_values(["analog_tier", "analog_score", "event_count"], ascending=[True, False, False]).head(max_candidates)
    controls = primitive_rows[
        primitive_rows["analog_tier"].astype(str).eq("source_preserved_multifragment_control_like")
        | (
            primitive_rows["endpoint_host_scope"].astype(str).eq("all_events_source_host_preserved")
            & ~primitive_rows["is_joint_weak_pair_analog_candidate"].astype(bool)
        )
    ].copy()
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        pool = controls[controls["branch"].astype(str).eq(str(candidate.branch))]
        if pool.empty:
            pool = controls
        if pool.empty:
            continue
        distances = (
            (pd.to_numeric(pool["event_count"], errors="coerce") - float(candidate.event_count)).abs()
            + (pd.to_numeric(pool["effective_segment_count_median"], errors="coerce") - float(candidate.effective_segment_count_median)).abs()
            + (pd.to_numeric(pool["top1_segment_share_ref_weight_median"], errors="coerce") - float(candidate.top1_segment_share_ref_weight_median)).abs()
        )
        matched = pool.loc[distances.sort_values().index[0]]
        rows.append(
            {
                "candidate_primitive_id": str(candidate.primitive_id),
                "candidate_source_family_id": str(candidate.source_family_id),
                "candidate_analog_tier": str(candidate.analog_tier),
                "candidate_branch": str(candidate.branch),
                "candidate_event_count": int(candidate.event_count),
                "candidate_effective_segment_count_median": float(candidate.effective_segment_count_median),
                "candidate_top1_segment_share_ref_weight_median": float(candidate.top1_segment_share_ref_weight_median),
                "candidate_host_context_class_mode": str(candidate.host_context_class_mode),
                "candidate_dominant_host_handle_id_mode_share": float(candidate.dominant_host_handle_id_mode_share),
                "control_primitive_id": str(matched["primitive_id"]),
                "control_source_family_id": str(matched["source_family_id"]),
                "control_analog_tier": str(matched["analog_tier"]),
                "control_branch": str(matched["branch"]),
                "control_event_count": int(matched["event_count"]),
                "control_effective_segment_count_median": float(matched["effective_segment_count_median"]),
                "control_top1_segment_share_ref_weight_median": float(matched["top1_segment_share_ref_weight_median"]),
                "control_host_context_class_mode": str(matched["host_context_class_mode"]),
                "control_dominant_host_handle_id_mode_share": float(matched["dominant_host_handle_id_mode_share"]),
                "matching_distance": float(distances.loc[matched.name]),
                "control_read": "nearest source-preserved or non-analog primitive by branch/event/effective/top1 screen metrics",
            }
        )
    return _with_claim_columns(pd.DataFrame(rows))


def _gate_matrix(primitive_rows: pd.DataFrame, source_rows: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    tier1_count = int(primitive_rows["analog_tier"].astype(str).eq("tier1_external_multi_fragment_host_competition_analog").sum())
    candidate_count = int(primitive_rows["is_joint_weak_pair_analog_candidate"].astype(bool).sum())
    source_candidate_count = int(source_rows["source_screen_status"].astype(str).ne("source_family_has_no_joint_analog_candidate").sum())
    residual_candidate_count = int(
        primitive_rows[
            primitive_rows["is_joint_weak_pair_analog_candidate"].astype(bool)
            & primitive_rows["residual_caveat_status"].astype(str).eq("source_family_has_residual_definition_debt")
        ].shape[0]
    )
    rows = [
        {
            "gate_id": "A1_input_surface_loaded",
            "gate_question": "Did the v2.2 measurement surface load with accepted primitive rows?",
            "status": "pass" if len(primitive_rows) > 0 else "blocked_missing_measurement_rows",
            "evidence": f"primitive_rows={len(primitive_rows)}",
            "decision": "analog screen is interpretable only if pass",
        },
        {
            "gate_id": "A2_joint_analog_candidates_exist",
            "gate_question": "Are there recurrent multi-fragment primitives matching the joint analog screen?",
            "status": "pass" if candidate_count > 0 else "blocked_no_joint_analog_candidates",
            "evidence": f"candidate_primitive_count={candidate_count}, tier1_count={tier1_count}",
            "decision": "advance to source-family review only if pass",
        },
        {
            "gate_id": "A3_source_family_support",
            "gate_question": "Do candidate primitives occur across source families rather than a single isolated row?",
            "status": "pass" if source_candidate_count >= 5 else "caveat_too_few_source_families",
            "evidence": f"candidate_source_family_count={source_candidate_count}",
            "decision": "external screen is stronger when candidates are not singleton artifacts",
        },
        {
            "gate_id": "A4_control_like_contrasts",
            "gate_question": "Are source-preserved or non-analog contrast rows available for candidate review?",
            "status": "pass" if len(controls) > 0 else "caveat_no_control_like_matches",
            "evidence": f"matched_control_rows={len(controls)}",
            "decision": "do not promote real-data analogy without contrast rows",
        },
        {
            "gate_id": "A5_residual_debt_visibility",
            "gate_question": "Are residual definition caveats explicitly counted for candidates?",
            "status": "pass",
            "evidence": f"candidate_rows_with_residual_debt={residual_candidate_count}",
            "decision": "residual debt must remain a caveat, not a failure hidden by the screen",
        },
        {
            "gate_id": "A6_claim_boundary_closed",
            "gate_question": "Does this screen avoid route/pathway, wall, quality/cost, and method claims?",
            "status": "closed_excluded_by_design",
            "evidence": CLAIM_BOUNDARY,
            "decision": "screen can justify local panel design, not real-data method success",
        },
    ]
    return _with_claim_columns(pd.DataFrame(rows))


def _readiness(gates: pd.DataFrame) -> str:
    statuses = set(gates["status"].astype(str))
    if any(status.startswith("blocked") for status in statuses):
        return "blocked_no_joint_analog_surface"
    if any(status.startswith("caveat") for status in statuses):
        return "caveated_analog_surface_review_only"
    return "ready_for_local_panel_design_review"


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
                values.append(str(value).replace("|", r"\|"))
        table.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        table.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(table)


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    primitive_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
    controls: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    candidates = primitive_rows[primitive_rows["is_joint_weak_pair_analog_candidate"].astype(bool)]
    lines = [
        "# NanoClustering Joint Weak-Pair Analog Screen",
        "",
        f"- measurement_dir: `{summary['measurement_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- readiness: `{summary['analog_screen_readiness']}`",
        f"- accepted_primitive_count: `{summary['accepted_primitive_count']}`",
        f"- analog_candidate_count: `{summary['analog_candidate_count']}`",
        f"- tier1_candidate_count: `{summary['tier1_candidate_count']}`",
        f"- candidate_source_family_count: `{summary['candidate_source_family_count']}`",
        f"- matched_control_count: `{summary['matched_control_count']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(gates, ["gate_id", "status", "evidence", "decision"], max_rows=12),
        "",
        "## Top Primitive Candidates",
        "",
        _markdown_table(
            candidates,
            [
                "primitive_id",
                "analog_tier",
                "analog_score",
                "event_count",
                "split_vector_class_mode",
                "host_context_class_mode",
                "effective_segment_count_median",
                "top1_segment_share_ref_weight_median",
                "dominant_host_handle_id_mode_share",
                "source_family_distinct_host_mode_count",
            ],
            max_rows=20,
        ),
        "",
        "## Source Families",
        "",
        _markdown_table(
            source_rows[source_rows["source_screen_status"].astype(str).ne("source_family_has_no_joint_analog_candidate")],
            [
                "source_family_id",
                "source_screen_status",
                "candidate_primitive_count",
                "tier1_candidate_count",
                "max_analog_score",
                "dominant_host_mode_count",
                "split_vector_mode_counts",
                "host_context_mode_counts",
            ],
            max_rows=20,
        ),
        "",
        "## Matched Controls",
        "",
        _markdown_table(
            controls,
            [
                "candidate_primitive_id",
                "candidate_analog_tier",
                "control_primitive_id",
                "control_analog_tier",
                "matching_distance",
                "control_read",
            ],
            max_rows=20,
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- This screen identifies analog structure in existing endpoint-vector measurements.",
        "- It does not prove that a real-data joint initialization will work.",
        "- The valid next use is local panel design, with matched controls and frozen pre-endpoint roles.",
    ]
    output_dir.joinpath(REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_screen(
    *,
    measurement_dir: Path,
    output_dir: Path,
    max_control_matches: int,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_JSON
    if summary_path.exists() and not force:
        raise FileExistsError(f"{_rel(summary_path)} already exists. Use --force to regenerate.")

    measurement_rows = _read_csv(measurement_dir / MEASUREMENT_ROWS_CSV)
    event_rows = _read_csv(measurement_dir / EVENT_ROWS_CSV)
    primitive_rows = _primitive_rows(measurement_rows, event_rows)
    source_rows = _source_family_rows(primitive_rows)
    controls = _match_controls(primitive_rows, max_candidates=max_control_matches)
    gates = _gate_matrix(primitive_rows, source_rows, controls)
    readiness = _readiness(gates)

    candidate_rows = primitive_rows[primitive_rows["is_joint_weak_pair_analog_candidate"].astype(bool)]
    tier1_rows = primitive_rows[
        primitive_rows["analog_tier"].astype(str).eq("tier1_external_multi_fragment_host_competition_analog")
    ]
    source_candidates = source_rows[
        source_rows["source_screen_status"].astype(str).ne("source_family_has_no_joint_analog_candidate")
    ]

    config = {
        "measurement_dir": _rel(measurement_dir),
        "output_dir": _rel(output_dir),
        "max_control_matches": max_control_matches,
        "analog_tier_rules": {
            "tier1": "event_count >= 5, effective >= 3, top1 < 0.5, split_count >= 3, external host, and host competition",
            "tier2": "event_count >= 5, effective >= 3, top1 < 0.5, split_count >= 3",
            "tier3": "event_count >= 3, effective >= 3, top1 < 0.5, split_count >= 3",
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output_dir.joinpath(CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(primitive_rows, output_dir / PRIMITIVE_ANALOG_ROWS_CSV)
    _write_csv(source_rows, output_dir / SOURCE_FAMILY_ROWS_CSV)
    _write_csv(controls, output_dir / MATCHED_CONTROL_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)

    summary = {
        "measurement_dir": _rel(measurement_dir),
        "output_dir": _rel(output_dir),
        "accepted_primitive_count": int(len(primitive_rows)),
        "accepted_source_family_count": int(primitive_rows["source_family_id"].nunique()),
        "analog_candidate_count": int(len(candidate_rows)),
        "tier1_candidate_count": int(len(tier1_rows)),
        "candidate_source_family_count": int(len(source_candidates)),
        "matched_control_count": int(len(controls)),
        "analog_tier_counts": {
            str(key): int(value)
            for key, value in primitive_rows["analog_tier"].value_counts().sort_index().to_dict().items()
        },
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gates["status"].value_counts().sort_index().to_dict().items()
        },
        "analog_screen_readiness": readiness,
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            PRIMITIVE_ANALOG_ROWS_CSV,
            SOURCE_FAMILY_ROWS_CSV,
            MATCHED_CONTROL_ROWS_CSV,
            GATE_MATRIX_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(
        output_dir=output_dir,
        summary=summary,
        primitive_rows=primitive_rows,
        source_rows=source_rows,
        controls=controls,
        gates=gates,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen NanoClustering v2.2 measurements for joint weak-pair analog structures."
    )
    parser.add_argument("--measurement-dir", type=Path, default=DEFAULT_MEASUREMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-control-matches", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_screen(
        measurement_dir=args.measurement_dir,
        output_dir=args.output_dir,
        max_control_matches=args.max_control_matches,
        force=args.force,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
