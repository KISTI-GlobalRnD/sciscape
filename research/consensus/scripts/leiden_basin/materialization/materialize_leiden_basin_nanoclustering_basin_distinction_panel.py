#!/usr/bin/env python3
"""Materialize a primitive NanoClustering basin-distinction panel.

This reads the definition-core endpoint-pair cases and separates observed
endpoint handles, pair relations, and family-level distinction classes. It is
membership-only endpoint cartography. It does not run clustering, execute
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
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_basin_distinction_panel_20260530"
)

PAIR_EVENT_ROWS_CSV = "nanoclustering_definition_core_pair_event_rows.csv"

HANDLE_ROWS_CSV = "nanoclustering_basin_distinction_handle_rows.csv"
RELATION_ROWS_CSV = "nanoclustering_basin_distinction_relation_rows.csv"
FAMILY_ROWS_CSV = "nanoclustering_basin_distinction_family_rows.csv"
ARCHETYPE_SUMMARY_CSV = "nanoclustering_basin_distinction_archetype_summary.csv"
SUMMARY_JSON = "nanoclustering_basin_distinction_summary.json"
REPORT_MD = "nanoclustering_basin_distinction_report.md"
CONFIG_JSON = "nanoclustering_basin_distinction_config.json"

CLAIM_BOUNDARY = (
    "Primitive basin-distinction endpoint cartography only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_basin_distinction_panel"

RELATION_AXIS_BY_PATTERN = {
    "split_and_merge_boundary": "fragmentation_with_host_merge",
    "severe_split_boundary": "severe_fragmentation",
    "merge_absorption_boundary": "host_absorption",
    "moderate_split_boundary": "moderate_fragmentation",
    "mild_or_label_reassignment_boundary": "mild_or_label_reassignment",
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


def _as_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _join_unique(values: pd.Series, *, limit: int = 20) -> str:
    seen: list[str] = []
    for value in values.dropna().astype(str):
        if value not in seen:
            seen.append(value)
    if len(seen) > limit:
        return ";".join(seen[:limit]) + f";...(+{len(seen) - limit})"
    return ";".join(seen)


def _count_string(values: pd.Series) -> str:
    counts = Counter(values.dropna().astype(str))
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _dominant_value(values: pd.Series) -> str:
    counts = Counter(values.dropna().astype(str))
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _source_handle_id(reference_run_id: str, ref_cluster_id: Any) -> str:
    return f"{reference_run_id}:ref{int(ref_cluster_id)}"


def _target_handle_id(comparison_run_id: str, best_run_cluster_id: Any) -> str:
    return f"{comparison_run_id}:run{int(best_run_cluster_id)}"


def _relation_axis(boundary_pattern: str) -> str:
    return RELATION_AXIS_BY_PATTERN.get(str(boundary_pattern), "unknown_endpoint_relation")


def _primitive_relation_status(row: pd.Series) -> str:
    axis = row["relation_axis"]
    top_split = _as_float(row.get("top_split_share_ref_weight"))
    target_share = _as_float(row.get("target_share_of_best_run_cluster_weight"))
    split_segments_ge5 = _as_int(row.get("split_segment_count_ge5_weight")) or 0
    merge_contributors_ge5 = _as_int(row.get("merge_contributor_count_ge5_weight")) or 0

    if axis in {"fragmentation_with_host_merge", "severe_fragmentation"}:
        if top_split is not None and top_split < 0.5 and split_segments_ge5 >= 2:
            return "accepted_primitive_distinct_endpoint_pair"
        return "fragmentation_candidate_missing_split_support"
    if axis == "host_absorption":
        if target_share is not None and target_share < 0.5 and merge_contributors_ge5 >= 2:
            return "accepted_absorption_distinct_endpoint_pair"
        return "absorption_candidate_missing_host_support"
    if axis == "moderate_fragmentation":
        if top_split is not None and top_split < 0.8:
            return "weak_boundary_candidate_needs_repetition"
        return "moderate_candidate_below_distinction_gate"
    return "not_distinct_under_primitive_gate"


def _is_accepted_status(status: str) -> bool:
    return status.startswith("accepted_")


def _family_distinction_class(group: pd.DataFrame) -> str:
    total = len(group)
    if total == 0:
        return "empty_family"
    split_like = int(
        group["relation_axis"].isin(
            {"fragmentation_with_host_merge", "severe_fragmentation"}
        ).sum()
    )
    absorption = int(group["relation_axis"].eq("host_absorption").sum())
    moderate = int(group["relation_axis"].eq("moderate_fragmentation").sum())
    accepted = int(group["is_accepted_distinct_relation"].sum())

    if accepted == 0:
        return "no_accepted_distinct_endpoint_relation"
    if split_like / total >= 0.67:
        return "fragmentation_dominant_multi_endpoint_family"
    if absorption / total >= 0.67:
        return "absorption_host_dominant_family"
    if split_like > 0 and absorption > 0:
        return "mixed_fragmentation_absorption_family"
    if moderate / total >= 0.5:
        return "moderate_fragmentation_family"
    return "heterogeneous_endpoint_boundary_family"


def _family_distinction_status(group: pd.DataFrame) -> str:
    accepted = int(group["is_accepted_distinct_relation"].sum())
    target_count = int(group["target_basin_handle_id"].nunique())
    if accepted >= 2 and target_count >= 2:
        return "multiple_observed_basin_candidates_v0"
    if accepted == 1:
        return "single_observed_alternative_basin_candidate_v0"
    return "not_distinct_under_current_primitive_gate"


def _relation_rows(pair_events: pd.DataFrame) -> pd.DataFrame:
    rows = pair_events.copy()
    rows["source_basin_handle_id"] = rows.apply(
        lambda row: _source_handle_id(row["reference_run_id"], row["ref_cluster_id"]),
        axis=1,
    )
    rows["target_basin_handle_id"] = rows.apply(
        lambda row: _target_handle_id(row["comparison_run_id"], row["best_run_cluster_id"]),
        axis=1,
    )
    rows["source_basin_role"] = "seed0_reference_endpoint_handle"
    rows["target_basin_role"] = "comparison_seed_endpoint_handle"
    rows["relation_axis"] = rows["boundary_pattern"].map(_relation_axis)
    rows["primitive_relation_status"] = rows.apply(_primitive_relation_status, axis=1)
    rows["is_accepted_distinct_relation"] = rows["primitive_relation_status"].map(
        _is_accepted_status
    )
    rows["basin_identity_scope"] = "endpoint_handle_local_to_run"
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
        "target_basin_handle_id",
        "source_basin_role",
        "target_basin_role",
        "primitive_relation_status",
        "is_accepted_distinct_relation",
        "relation_axis",
        "boundary_pattern",
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
        "top_split_share_ref_weight",
        "fragmentation_index",
        "split_segment_count",
        "split_segment_count_ge5_weight",
        "merge_contributor_count",
        "merge_contributor_count_ge5_weight",
        "target_share_of_best_run_cluster_weight",
        "best_run_cluster_weight_sum",
        "best_run_cluster_unit_count",
        "basin_identity_scope",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(
        [
            "branch",
            "boundary_family_tier",
            "family_id",
            "comparison_seed",
            "event_id",
        ]
    )


def _handle_rows(relations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for handle_id, group in relations.groupby("source_basin_handle_id", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "basin_handle_id": handle_id,
                "basin_handle_role": "seed0_reference_endpoint_handle",
                "candidate_status": "source_reference_observed_basin_candidate_v0",
                "branch": first["branch"],
                "endpoint_run_id": first["reference_run_id"],
                "endpoint_seed": int(first["reference_seed"]),
                "endpoint_cluster_id": int(first["ref_cluster_id"]),
                "source_family_count": int(group["family_id"].nunique()),
                "relation_event_count": int(len(group)),
                "accepted_relation_event_count": int(group["is_accepted_distinct_relation"].sum()),
                "source_family_ids": _join_unique(group["family_id"]),
                "source_ref_cluster_ids": _join_unique(group["ref_cluster_id"]),
                "target_handle_count": int(group["target_basin_handle_id"].nunique()),
                "relation_axis_counts": _count_string(group["relation_axis"]),
                "boundary_pattern_counts": _count_string(group["boundary_pattern"]),
                "dominant_relation_axis": _dominant_value(group["relation_axis"]),
                "boundary_family_tiers": _join_unique(group["boundary_family_tier"]),
                "ref_unit_count_min": int(group["ref_unit_count"].min()),
                "ref_weight_sum_min": float(group["ref_weight_sum"].min()),
                "top_split_share_ref_weight_min": float(group["top_split_share_ref_weight"].min()),
                "top_split_share_ref_weight_median": float(
                    group["top_split_share_ref_weight"].median()
                ),
                "handle_reuse_status": "source_handle",
                "basin_identity_scope": "endpoint_handle_local_to_run",
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    for handle_id, group in relations.groupby("target_basin_handle_id", sort=False):
        first = group.iloc[0]
        accepted_count = int(group["is_accepted_distinct_relation"].sum())
        source_count = int(group["family_id"].nunique())
        rows.append(
            {
                "basin_handle_id": handle_id,
                "basin_handle_role": "comparison_seed_endpoint_handle",
                "candidate_status": (
                    "comparison_observed_basin_candidate_v0"
                    if accepted_count > 0
                    else "comparison_endpoint_below_distinction_gate"
                ),
                "branch": first["branch"],
                "endpoint_run_id": first["comparison_run_id"],
                "endpoint_seed": int(first["comparison_seed"]),
                "endpoint_cluster_id": int(first["best_run_cluster_id"]),
                "source_family_count": source_count,
                "relation_event_count": int(len(group)),
                "accepted_relation_event_count": accepted_count,
                "source_family_ids": _join_unique(group["family_id"]),
                "source_ref_cluster_ids": _join_unique(group["ref_cluster_id"]),
                "target_handle_count": 1,
                "relation_axis_counts": _count_string(group["relation_axis"]),
                "boundary_pattern_counts": _count_string(group["boundary_pattern"]),
                "dominant_relation_axis": _dominant_value(group["relation_axis"]),
                "boundary_family_tiers": _join_unique(group["boundary_family_tier"]),
                "ref_unit_count_min": int(group["ref_unit_count"].min()),
                "ref_weight_sum_min": float(group["ref_weight_sum"].min()),
                "top_split_share_ref_weight_min": float(group["top_split_share_ref_weight"].min()),
                "top_split_share_ref_weight_median": float(
                    group["top_split_share_ref_weight"].median()
                ),
                "handle_reuse_status": (
                    "shared_target_handle_across_sources"
                    if source_count > 1
                    else "single_source_target_handle"
                ),
                "basin_identity_scope": "endpoint_handle_local_to_run",
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["branch", "basin_handle_role", "source_family_count", "relation_event_count"],
        ascending=[True, True, False, False],
    )


def _family_rows(relations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id, group in relations.groupby("family_id", sort=False):
        first = group.iloc[0]
        accepted = int(group["is_accepted_distinct_relation"].sum())
        rows.append(
            {
                "family_id": family_id,
                "branch": first["branch"],
                "ref_cluster_id": int(first["ref_cluster_id"]),
                "boundary_family_tier": first["boundary_family_tier"],
                "definition_readiness": first["definition_readiness"],
                "source_basin_handle_id": first["source_basin_handle_id"],
                "observed_basin_candidate_count_v0": int(
                    1 + group["target_basin_handle_id"].nunique()
                ),
                "comparison_endpoint_handle_count": int(
                    group["target_basin_handle_id"].nunique()
                ),
                "relation_event_count": int(len(group)),
                "accepted_distinct_relation_count": accepted,
                "accepted_distinct_relation_share": accepted / len(group) if len(group) else 0.0,
                "severe_endpoint_pair_count": int(
                    group["endpoint_pair_role"].astype(str).str.startswith("severe_").sum()
                ),
                "relation_axis_counts": _count_string(group["relation_axis"]),
                "boundary_pattern_counts": _count_string(group["boundary_pattern"]),
                "basin_distinction_class": _family_distinction_class(group),
                "basin_distinction_status": _family_distinction_status(group),
                "comparison_seeds": _join_unique(group["comparison_seed"]),
                "target_basin_handle_ids": _join_unique(group["target_basin_handle_id"], limit=30),
                "top_split_share_ref_weight_min": float(group["top_split_share_ref_weight"].min()),
                "top_split_share_ref_weight_median": float(
                    group["top_split_share_ref_weight"].median()
                ),
                "fragmentation_index_median": float(group["fragmentation_index"].median()),
                "split_segment_count_ge5_weight_max": int(
                    group["split_segment_count_ge5_weight"].max()
                ),
                "merge_contributor_count_ge5_weight_max": int(
                    group["merge_contributor_count_ge5_weight"].max()
                ),
                "target_share_of_best_run_cluster_weight_median": float(
                    group["target_share_of_best_run_cluster_weight"].median()
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "branch",
            "boundary_family_tier",
            "basin_distinction_class",
            "accepted_distinct_relation_count",
            "family_id",
        ],
        ascending=[True, True, True, False, True],
    )


def _archetype_summary(family_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (tier, distinction_class), group in family_rows.groupby(
        ["boundary_family_tier", "basin_distinction_class"], sort=True
    ):
        rows.append(
            {
                "boundary_family_tier": tier,
                "basin_distinction_class": distinction_class,
                "family_count": int(len(group)),
                "observed_basin_candidate_count_v0_sum": int(
                    group["observed_basin_candidate_count_v0"].sum()
                ),
                "comparison_endpoint_handle_count_sum": int(
                    group["comparison_endpoint_handle_count"].sum()
                ),
                "accepted_distinct_relation_count_sum": int(
                    group["accepted_distinct_relation_count"].sum()
                ),
                "top_split_share_ref_weight_min": float(
                    group["top_split_share_ref_weight_min"].min()
                ),
                "fragmentation_index_median": float(group["fragmentation_index_median"].median()),
                "status": "primitive_basin_distinction_summary",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["boundary_family_tier", "basin_distinction_class"]
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
    suffix = []
    if len(frame) > max_rows:
        suffix.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    handle_rows: pd.DataFrame,
    relation_rows: pd.DataFrame,
    family_rows: pd.DataFrame,
    archetype_summary: pd.DataFrame,
) -> None:
    accepted_relation_count = int(relation_rows["is_accepted_distinct_relation"].sum())
    shared_target_count = int(
        handle_rows["handle_reuse_status"].eq("shared_target_handle_across_sources").sum()
    )
    text = [
        "# NanoClustering Basin Distinction Panel",
        "",
        f"- handle_rows: `{len(handle_rows)}`",
        f"- relation_rows: `{len(relation_rows)}`",
        f"- family_rows: `{len(family_rows)}`",
        f"- archetype_summary_rows: `{len(archetype_summary)}`",
        f"- accepted_distinct_relation_count: `{accepted_relation_count}`",
        f"- shared_target_handle_count: `{shared_target_count}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Family Distinction Classes",
        "",
        _markdown_table(
            archetype_summary,
            [
                "boundary_family_tier",
                "basin_distinction_class",
                "family_count",
                "observed_basin_candidate_count_v0_sum",
                "accepted_distinct_relation_count_sum",
            ],
            max_rows=20,
        ),
        "",
        "## Family Rows",
        "",
        _markdown_table(
            family_rows,
            [
                "family_id",
                "boundary_family_tier",
                "observed_basin_candidate_count_v0",
                "accepted_distinct_relation_count",
                "basin_distinction_class",
                "relation_axis_counts",
                "boundary_pattern_counts",
            ],
            max_rows=25,
        ),
        "",
        "## Handle Reuse",
        "",
        _markdown_table(
            handle_rows[handle_rows["source_family_count"].gt(1)],
            [
                "basin_handle_id",
                "basin_handle_role",
                "candidate_status",
                "source_family_count",
                "relation_event_count",
                "dominant_relation_axis",
                "source_family_ids",
            ],
            max_rows=20,
        ),
        "",
        "## Read",
        "",
        "- A primitive basin candidate is an endpoint handle, not a final global attraction basin.",
        "- Each definition-core family supplies one seed0 reference handle plus comparison-seed endpoint handles.",
        "- Relation rows distinguish fragmentation, host absorption, and mixed endpoint-boundary archetypes without using quality/cost or pathway traces.",
        "- The panel is ready for internal basin-family coherence checks, but it does not support wall/pathway or basin-quality claims.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(*, pair_case_dir: Path, output_dir: Path) -> dict[str, Any]:
    pair_events = _read_csv(pair_case_dir / PAIR_EVENT_ROWS_CSV)
    relation_rows = _relation_rows(pair_events)
    handle_rows = _handle_rows(relation_rows)
    family_rows = _family_rows(relation_rows)
    archetype_summary = _archetype_summary(family_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(handle_rows, output_dir / HANDLE_ROWS_CSV)
    _write_csv(relation_rows, output_dir / RELATION_ROWS_CSV)
    _write_csv(family_rows, output_dir / FAMILY_ROWS_CSV)
    _write_csv(archetype_summary, output_dir / ARCHETYPE_SUMMARY_CSV)
    _write_report(
        output_dir=output_dir,
        handle_rows=handle_rows,
        relation_rows=relation_rows,
        family_rows=family_rows,
        archetype_summary=archetype_summary,
    )

    summary = {
        "ok": True,
        "pair_case_dir": _rel(pair_case_dir),
        "output_dir": _rel(output_dir),
        "handle_row_count": int(len(handle_rows)),
        "relation_row_count": int(len(relation_rows)),
        "family_row_count": int(len(family_rows)),
        "archetype_summary_row_count": int(len(archetype_summary)),
        "accepted_distinct_relation_count": int(
            relation_rows["is_accepted_distinct_relation"].sum()
        ),
        "source_handle_count": int(
            handle_rows["basin_handle_role"].eq("seed0_reference_endpoint_handle").sum()
        ),
        "target_handle_count": int(
            handle_rows["basin_handle_role"].eq("comparison_seed_endpoint_handle").sum()
        ),
        "shared_target_handle_count": int(
            handle_rows["handle_reuse_status"].eq("shared_target_handle_across_sources").sum()
        ),
        "family_distinction_class_counts": {
            str(key): int(value)
            for key, value in family_rows["basin_distinction_class"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "handle_rows_csv": _rel(output_dir / HANDLE_ROWS_CSV),
            "relation_rows_csv": _rel(output_dir / RELATION_ROWS_CSV),
            "family_rows_csv": _rel(output_dir / FAMILY_ROWS_CSV),
            "archetype_summary_csv": _rel(output_dir / ARCHETYPE_SUMMARY_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "pair_case_dir": _rel(pair_case_dir),
        "output_dir": _rel(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
        "primitive_relation_statuses": {
            "fragmentation_with_host_merge": "accepted when top_split_share_ref_weight < 0.5 and split_segment_count_ge5_weight >= 2",
            "severe_fragmentation": "accepted when top_split_share_ref_weight < 0.5 and split_segment_count_ge5_weight >= 2",
            "host_absorption": "accepted when target_share_of_best_run_cluster_weight < 0.5 and merge_contributor_count_ge5_weight >= 2",
            "moderate_fragmentation": "weak candidate, not accepted distinct, when top_split_share_ref_weight < 0.8",
        },
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        pair_case_dir=args.pair_case_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
