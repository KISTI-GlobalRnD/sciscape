#!/usr/bin/env python3
"""Materialize the NanoClustering definition-core v1 family registry.

This reads the full definition-core basin-vector coherence panel and assigns a
registry status to each support-local endpoint-vector family. It does not run
clustering, execute optimizer routes, promote wall/pathway claims, or inspect
basin quality/cost.
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
DEFAULT_PAIR_CASE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_definition_core_full_pair_cases_20260530"
)
DEFAULT_COHERENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_full_basin_vector_coherence_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v1_family_registry_20260530"
)

SELECTED_FAMILIES_CSV = "nanoclustering_definition_core_selected_families.csv"
COHERENCE_FAMILY_ROWS_CSV = "nanoclustering_basin_vector_coherence_family_rows.csv"

V1_FAMILY_REGISTRY_CSV = "nanoclustering_definition_core_v1_family_registry.csv"
V1_STATUS_SUMMARY_CSV = "nanoclustering_definition_core_v1_status_summary.csv"
V1_TIER_SUMMARY_CSV = "nanoclustering_definition_core_v1_tier_summary.csv"
V1_CANDIDATE_SHORTLIST_CSV = "nanoclustering_definition_core_v1_candidate_shortlist.csv"
SUMMARY_JSON = "nanoclustering_definition_core_v1_family_registry_summary.json"
REPORT_MD = "nanoclustering_definition_core_v1_family_registry_report.md"
CONFIG_JSON = "nanoclustering_definition_core_v1_family_registry_config.json"

CLAIM_BOUNDARY = (
    "Definition-core v1 family registry only; no route execution, wall/pathway "
    "promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_definition_core_v1_registry"


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


def _v1_registry_status(coherence_status: str) -> str:
    if coherence_status == "coherent_vector_and_host_family":
        return "definition_core_v1_coherent"
    if coherence_status == "coherent_class_with_numeric_variation":
        return "definition_core_v1_numeric_stress"
    if coherence_status == "split_coherent_host_variable_family":
        return "split_coherent_host_variable_subfamily"
    if coherence_status == "host_coherent_split_mixed_family":
        return "host_coherent_split_mixed_subfamily"
    return "heterogeneous_rule_edge_review"


def _v1_read(row: pd.Series) -> str:
    status = str(row["definition_core_v1_status"])
    if status == "definition_core_v1_coherent":
        return "accepted support-local endpoint-vector family for definition-core v1"
    if status == "definition_core_v1_numeric_stress":
        return "class and host repeat, but numeric split bands vary beyond strict stability"
    if status == "split_coherent_host_variable_subfamily":
        return "split shape repeats, but host context changes; split by host context before promotion"
    if status == "host_coherent_split_mixed_subfamily":
        return "host context repeats, but split shape or class changes; split by shape core before promotion"
    return "heterogeneous family or current vector rule edge; review before promotion"


def _next_definition_action(row: pd.Series) -> str:
    status = str(row["definition_core_v1_status"])
    if status == "definition_core_v1_coherent":
        return "retain_as_coherent_definition_core_v1_family"
    if status == "definition_core_v1_numeric_stress":
        return "hold_for_numeric_band_stability_check"
    if status == "split_coherent_host_variable_subfamily":
        return "partition_by_dominant_host_context"
    if status == "host_coherent_split_mixed_subfamily":
        return "partition_by_shape_core_or_boundary_pattern"
    return "review_vector_rule_or_split_into_smaller_families"


def _registry_rows(*, selected_families: pd.DataFrame, coherence_rows: pd.DataFrame) -> pd.DataFrame:
    family_meta_cols = [
        "family_id",
        "family_selection_scope",
        "panel_rank_in_branch_tier",
        "panel_selection_reason",
        "branch",
        "ref_cluster_id",
        "boundary_family_tier",
        "definition_readiness",
        "expected_archetype_from_current_panel",
        "ref_unit_count",
        "ref_weight_sum",
        "strong_seed_count",
        "severe_seed_count",
        "top_split_share_min",
        "top_split_share_median",
        "strong_seed_list",
        "severe_seed_list",
    ]
    rows = coherence_rows.merge(
        selected_families[family_meta_cols],
        on=["family_id", "branch", "boundary_family_tier"],
        how="left",
        validate="one_to_one",
    )
    if rows["ref_weight_sum"].isna().any():
        raise ValueError("coherence row missing selected-family metadata")
    rows["definition_core_v1_status"] = rows["coherence_status"].map(_v1_registry_status)
    rows["definition_core_v1_read"] = rows.apply(_v1_read, axis=1)
    rows["next_definition_action"] = rows.apply(_next_definition_action, axis=1)
    rows["definition_core_v1_acceptance_status"] = rows["definition_core_v1_status"].map(
        lambda status: (
            "accepted_endpoint_vector_family"
            if status == "definition_core_v1_coherent"
            else "not_accepted_requires_definition_refinement"
        )
    )
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "family_id",
        "branch",
        "ref_cluster_id",
        "boundary_family_tier",
        "definition_readiness",
        "definition_core_v1_status",
        "definition_core_v1_acceptance_status",
        "coherence_status",
        "family_vector_class",
        "event_count",
        "ref_unit_count",
        "ref_weight_sum",
        "strong_seed_count",
        "severe_seed_count",
        "top_split_share_min",
        "top_split_share_median",
        "dominant_split_vector_class",
        "dominant_split_vector_class_share",
        "dominant_host_context_class",
        "dominant_host_context_class_share",
        "dominant_shape_core_signature_share",
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
        "definition_core_v1_read",
        "next_definition_action",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(
        [
            "definition_core_v1_acceptance_status",
            "boundary_family_tier",
            "family_vector_class",
            "ref_weight_sum",
            "family_id",
        ],
        ascending=[True, True, True, False, True],
    )


def _status_summary(registry: pd.DataFrame) -> pd.DataFrame:
    rows = (
        registry.groupby(["definition_core_v1_status", "family_vector_class"], as_index=False)
        .agg(
            family_count=("family_id", "size"),
            event_count_sum=("event_count", "sum"),
            ref_weight_sum=("ref_weight_sum", "sum"),
            median_event_count=("event_count", "median"),
            median_ref_weight_sum=("ref_weight_sum", "median"),
            median_split_class_share=("dominant_split_vector_class_share", "median"),
            median_host_context_share=("dominant_host_context_class_share", "median"),
            median_shape_core_share=("dominant_shape_core_signature_share", "median"),
            median_host_handle_share=("dominant_host_handle_share", "median"),
        )
        .sort_values(["definition_core_v1_status", "family_count"], ascending=[True, False])
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _tier_summary(registry: pd.DataFrame) -> pd.DataFrame:
    rows = (
        registry.groupby(["boundary_family_tier", "definition_core_v1_status"], as_index=False)
        .agg(
            family_count=("family_id", "size"),
            event_count_sum=("event_count", "sum"),
            ref_weight_sum=("ref_weight_sum", "sum"),
            median_event_count=("event_count", "median"),
            median_top_split_share_min=("top_split_share_min", "median"),
            median_top1_segment_share=("top1_segment_share_median", "median"),
            median_effective_segment_count=("effective_segment_count_median", "median"),
        )
        .sort_values(["boundary_family_tier", "definition_core_v1_status"])
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _candidate_shortlist(registry: pd.DataFrame) -> pd.DataFrame:
    rows = registry[
        registry["definition_core_v1_status"].eq("definition_core_v1_coherent")
    ].copy()
    rows["coherence_rank_score"] = (
        rows["dominant_split_vector_class_share"].astype(float)
        + rows["dominant_host_context_class_share"].astype(float)
        + rows["dominant_shape_core_signature_share"].astype(float)
        + rows["dominant_host_handle_share"].astype(float)
        - rows["top1_segment_share_iqr"].astype(float)
        - rows["top2_segment_share_iqr"].astype(float)
    )
    return rows.sort_values(
        [
            "boundary_family_tier",
            "family_vector_class",
            "coherence_rank_score",
            "event_count",
            "ref_weight_sum",
        ],
        ascending=[True, True, False, False, False],
    ).head(40)


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
    registry: pd.DataFrame,
    status_summary: pd.DataFrame,
    tier_summary: pd.DataFrame,
    candidate_shortlist: pd.DataFrame,
) -> None:
    accepted = registry[
        registry["definition_core_v1_status"].eq("definition_core_v1_coherent")
    ]
    text = [
        "# NanoClustering Definition-Core V1 Family Registry",
        "",
        f"- family_rows: `{len(registry)}`",
        f"- accepted_endpoint_vector_families: `{len(accepted)}`",
        f"- nonaccepted_definition_refinement_families: `{len(registry) - len(accepted)}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## V1 Status Summary",
        "",
        _markdown_table(
            status_summary,
            [
                "definition_core_v1_status",
                "family_vector_class",
                "family_count",
                "event_count_sum",
                "ref_weight_sum",
                "median_split_class_share",
                "median_host_context_share",
                "median_shape_core_share",
                "median_host_handle_share",
            ],
            max_rows=30,
        ),
        "",
        "## Tier Summary",
        "",
        _markdown_table(
            tier_summary,
            [
                "boundary_family_tier",
                "definition_core_v1_status",
                "family_count",
                "event_count_sum",
                "ref_weight_sum",
                "median_top_split_share_min",
                "median_top1_segment_share",
                "median_effective_segment_count",
            ],
            max_rows=20,
        ),
        "",
        "## Coherent Candidate Shortlist",
        "",
        _markdown_table(
            candidate_shortlist,
            [
                "family_id",
                "boundary_family_tier",
                "family_vector_class",
                "event_count",
                "ref_weight_sum",
                "dominant_split_vector_class_share",
                "dominant_host_context_class_share",
                "dominant_shape_core_signature_share",
                "dominant_host_handle_share",
                "top1_segment_share_iqr",
                "next_definition_action",
            ],
            max_rows=30,
        ),
        "",
        "## Read",
        "",
        "- `definition_core_v1_coherent` is the current accepted primitive basin family status.",
        "- Numeric-stress, split-coherent host-variable, host-coherent split-mixed, and heterogeneous rows are not failures; they are definition-refinement work queues.",
        "- This registry remains membership-derived endpoint cartography and does not establish final global basins, wall/pathway traversal, quality, or cost claims.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(*, pair_case_dir: Path, coherence_dir: Path, output_dir: Path) -> dict[str, Any]:
    selected_families = _read_csv(pair_case_dir / SELECTED_FAMILIES_CSV)
    coherence_rows = _read_csv(coherence_dir / COHERENCE_FAMILY_ROWS_CSV)
    registry = _registry_rows(
        selected_families=selected_families,
        coherence_rows=coherence_rows,
    )
    status_summary = _status_summary(registry)
    tier_summary = _tier_summary(registry)
    candidate_shortlist = _candidate_shortlist(registry)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(registry, output_dir / V1_FAMILY_REGISTRY_CSV)
    _write_csv(status_summary, output_dir / V1_STATUS_SUMMARY_CSV)
    _write_csv(tier_summary, output_dir / V1_TIER_SUMMARY_CSV)
    _write_csv(candidate_shortlist, output_dir / V1_CANDIDATE_SHORTLIST_CSV)
    _write_report(
        output_dir=output_dir,
        registry=registry,
        status_summary=status_summary,
        tier_summary=tier_summary,
        candidate_shortlist=candidate_shortlist,
    )

    summary = {
        "ok": True,
        "pair_case_dir": _rel(pair_case_dir),
        "coherence_dir": _rel(coherence_dir),
        "output_dir": _rel(output_dir),
        "family_row_count": int(len(registry)),
        "accepted_endpoint_vector_family_count": int(
            registry["definition_core_v1_status"].eq("definition_core_v1_coherent").sum()
        ),
        "nonaccepted_definition_refinement_family_count": int(
            registry["definition_core_v1_status"].ne("definition_core_v1_coherent").sum()
        ),
        "definition_core_v1_status_counts": _count(registry, "definition_core_v1_status"),
        "tier_status_counts": {
            f"{tier}|{status}": int(count)
            for (tier, status), count in registry.groupby(
                ["boundary_family_tier", "definition_core_v1_status"]
            ).size().sort_index().to_dict().items()
        },
        "family_vector_class_counts": _count(registry, "family_vector_class"),
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "family_registry_csv": _rel(output_dir / V1_FAMILY_REGISTRY_CSV),
            "status_summary_csv": _rel(output_dir / V1_STATUS_SUMMARY_CSV),
            "tier_summary_csv": _rel(output_dir / V1_TIER_SUMMARY_CSV),
            "candidate_shortlist_csv": _rel(output_dir / V1_CANDIDATE_SHORTLIST_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "pair_case_dir": _rel(pair_case_dir),
        "coherence_dir": _rel(coherence_dir),
        "output_dir": _rel(output_dir),
        "accepted_status": "definition_core_v1_coherent",
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
    parser.add_argument("--coherence-dir", type=Path, default=DEFAULT_COHERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        pair_case_dir=args.pair_case_dir.resolve(),
        coherence_dir=args.coherence_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
