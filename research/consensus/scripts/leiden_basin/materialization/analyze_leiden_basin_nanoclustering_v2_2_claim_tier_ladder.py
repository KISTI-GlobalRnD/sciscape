#!/usr/bin/env python3
"""Build a claim-tier ladder for the v2.2 accepted primitive set.

The distribution review separates the 223 accepted primitives into broad
descriptive bands. This script turns those bands into an explicit cumulative
claim ladder:

T1 stable nucleus -> T2 thin-clean extension -> T3 thin concentration
caveat -> T4 concentration caveat -> T5 standard residual-debt caveat -> T6
high residual-debt priority.

The ladder is descriptive. It does not change v2.2 primitive membership, run
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
DEFAULT_DISTRIBUTION_REVIEW_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_v2_2_measurement_distribution_review_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_v2_2_claim_tier_ladder_20260531"
)

DISTRIBUTION_PRIMITIVE_ROWS_CSV = (
    "nanoclustering_v2_2_distribution_review_primitive_rows.csv"
)
RESIDUAL_DEBT_PRIORITY_ROWS_CSV = (
    "nanoclustering_v2_2_distribution_review_residual_debt_priority_rows.csv"
)
DISTRIBUTION_REVIEW_SUMMARY_JSON = "nanoclustering_v2_2_distribution_review_summary.json"

CLAIM_TIER_PRIMITIVE_ROWS_CSV = "nanoclustering_v2_2_claim_tier_primitive_rows.csv"
CLAIM_TIER_SUMMARY_CSV = "nanoclustering_v2_2_claim_tier_summary.csv"
CLAIM_TIER_CUMULATIVE_LADDER_CSV = "nanoclustering_v2_2_claim_tier_cumulative_ladder.csv"
CLAIM_TIER_HOST_BOUNDARY_SUMMARY_CSV = "nanoclustering_v2_2_claim_tier_host_boundary_summary.csv"
CLAIM_TIER_GATE_MATRIX_CSV = "nanoclustering_v2_2_claim_tier_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_v2_2_claim_tier_ladder_summary.json"
REPORT_MD = "nanoclustering_v2_2_claim_tier_ladder_report.md"
CONFIG_JSON = "nanoclustering_v2_2_claim_tier_ladder_config.json"

STABLE_BAND = "stable_high_support_measurement_unit"
THIN_CLEAN_BAND = "thin_clean_measurement_unit"
RESIDUAL_BAND = "residual_debt_caveated_measurement_unit"
CONCENTRATION_BAND = "concentration_caveated_measurement_unit"

CLAIM_BOUNDARY = (
    "V2.2 claim-tier ladder only; descriptive ordering of accepted primitives, "
    "no route execution, wall/pathway promotion, basin-quality claim, cost "
    "claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_v2_2_claim_tier_ladder"

TIER_ORDER = [
    "T1_stable_high_support_nucleus",
    "T2_thin_clean_extension",
    "T3_thin_concentration_caveat",
    "T4_concentration_caveat_no_residual",
    "T5_standard_residual_debt_caveat",
    "T6_high_residual_debt_priority",
]

TIER_READS = {
    "T1_stable_high_support_nucleus": (
        "Primary descriptive nucleus: enough support, no residual debt, and "
        "host/shape/boundary concentration above the review floor."
    ),
    "T2_thin_clean_extension": (
        "Clean extension: no residual debt and no concentration caveat, but "
        "support is thin."
    ),
    "T3_thin_concentration_caveat": (
        "Thin extension with additional concentration caveat; accepted but not "
        "core evidence."
    ),
    "T4_concentration_caveat_no_residual": (
        "Non-residual accepted primitive with moderate/deep support but weaker "
        "host, shape, or boundary concentration."
    ),
    "T5_standard_residual_debt_caveat": (
        "Accepted primitive from a source family with residual definition debt; "
        "use as caveated support, not as clean definition evidence."
    ),
    "T6_high_residual_debt_priority": (
        "Accepted primitive from a high residual-debt source family; keep as an "
        "audit priority before using in headline claims."
    ),
}

TIER_HEADLINE_USE = {
    "T1_stable_high_support_nucleus": "headline_nucleus",
    "T2_thin_clean_extension": "secondary_clean_extension",
    "T3_thin_concentration_caveat": "appendix_or_caveated_extension",
    "T4_concentration_caveat_no_residual": "mechanism_heterogeneity_caveat",
    "T5_standard_residual_debt_caveat": "residual_debt_caveat",
    "T6_high_residual_debt_priority": "audit_priority_not_headline",
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


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = frame.copy()
    for column in columns:
        if column in rows:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def _reason_tokens(value: Any) -> set[str]:
    return {token for token in str(value).split(";") if token and token != "none"}


def _claim_tier(row: pd.Series, high_residual_families: set[str]) -> str:
    if row["distribution_review_band"] == STABLE_BAND:
        return "T1_stable_high_support_nucleus"
    if str(row["source_family_id"]) in high_residual_families:
        return "T6_high_residual_debt_priority"
    if row["distribution_review_band"] == RESIDUAL_BAND:
        return "T5_standard_residual_debt_caveat"
    if row["distribution_review_band"] == CONCENTRATION_BAND:
        return "T4_concentration_caveat_no_residual"
    if row["distribution_review_band"] == THIN_CLEAN_BAND:
        extra_reasons = _reason_tokens(row["distribution_caveat_reasons"]) - {"thin_support"}
        if extra_reasons:
            return "T3_thin_concentration_caveat"
        return "T2_thin_clean_extension"
    raise ValueError(f"unclassified row: {row['primitive_id']}")


def _claim_tier_rows(
    *,
    review_rows: pd.DataFrame,
    residual_priority_rows: pd.DataFrame,
) -> pd.DataFrame:
    high_residual_families = set(
        residual_priority_rows.loc[
            residual_priority_rows["distribution_review_priority"].eq(
                "high_residual_debt_review"
            ),
            "source_family_id",
        ].astype(str)
    )
    rows = _numeric(
        review_rows,
        [
            "event_count",
            "source_family_residual_event_count",
            "dominant_host_handle_id_mode_share",
            "shape_core_signature_mode_share",
            "boundary_pattern_mode_share",
            "top1_segment_share_ref_weight_median",
            "effective_segment_count_median",
            "fragmentation_index_median",
        ],
    )
    rows["claim_tier"] = rows.apply(_claim_tier, axis=1, high_residual_families=high_residual_families)
    rows["claim_tier_rank"] = rows["claim_tier"].map(
        {tier: index + 1 for index, tier in enumerate(TIER_ORDER)}
    )
    rows["claim_tier_read"] = rows["claim_tier"].map(TIER_READS)
    rows["headline_use"] = rows["claim_tier"].map(TIER_HEADLINE_USE)
    rows["claim_ladder_status"] = "descriptive_ladder_only_not_definition_change"
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "claim_tier_rank",
        "claim_tier",
        "headline_use",
        "primitive_id",
        "source_family_id",
        "branch",
        "boundary_family_tier",
        "primitive_type",
        "measurement_support_class",
        "event_count",
        "distribution_review_band",
        "distribution_caveat_reasons",
        "host_context_class_mode",
        "boundary_pattern_mode",
        "dominant_host_handle_id_mode_share",
        "shape_core_signature_mode_share",
        "boundary_pattern_mode_share",
        "source_family_residual_event_count",
        "residual_queue_statuses",
        "top1_segment_share_ref_weight_median",
        "effective_segment_count_median",
        "fragmentation_index_median",
        "claim_tier_read",
        "claim_ladder_status",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    return rows[[column for column in preferred if column in rows.columns]].sort_values(
        [
            "claim_tier_rank",
            "boundary_family_tier",
            "host_context_class_mode",
            "boundary_pattern_mode",
            "source_family_id",
            "primitive_id",
        ]
    )


def _tier_summary(tier_rows: pd.DataFrame) -> pd.DataFrame:
    rows = (
        tier_rows.groupby(["claim_tier_rank", "claim_tier", "headline_use"], as_index=False)
        .agg(
            primitive_count=("primitive_id", "nunique"),
            event_count=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
            persistent_mixed_core_primitives=(
                "boundary_family_tier",
                lambda s: int((s == "persistent_mixed_core").sum()),
            ),
            repeat_severe_core_primitives=(
                "boundary_family_tier",
                lambda s: int((s == "repeat_severe_core").sum()),
            ),
            median_host_handle_mode_share=("dominant_host_handle_id_mode_share", "median"),
            median_shape_core_mode_share=("shape_core_signature_mode_share", "median"),
            median_boundary_pattern_mode_share=("boundary_pattern_mode_share", "median"),
            median_fragmentation_index=("fragmentation_index_median", "median"),
        )
        .sort_values("claim_tier_rank")
    )
    rows["claim_tier_read"] = rows["claim_tier"].map(TIER_READS)
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _cumulative_ladder(tier_rows: pd.DataFrame, tier_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cumulative_primitives = 0
    cumulative_events = 0
    cumulative_families: set[str] = set()
    for _, tier in tier_summary.sort_values("claim_tier_rank").iterrows():
        group = tier_rows[tier_rows["claim_tier"].eq(tier["claim_tier"])]
        cumulative_primitives += int(group["primitive_id"].nunique())
        cumulative_events += int(group["event_count"].sum())
        cumulative_families.update(group["source_family_id"].astype(str))
        rows.append(
            {
                "claim_tier_rank": int(tier["claim_tier_rank"]),
                "claim_tier": tier["claim_tier"],
                "incremental_primitive_count": int(group["primitive_id"].nunique()),
                "incremental_event_count": int(group["event_count"].sum()),
                "incremental_source_family_count": int(group["source_family_id"].nunique()),
                "cumulative_primitive_count": cumulative_primitives,
                "cumulative_event_count": cumulative_events,
                "cumulative_source_family_count": len(cumulative_families),
                "headline_use": tier["headline_use"],
                "claim_tier_read": tier["claim_tier_read"],
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _host_boundary_summary(tier_rows: pd.DataFrame) -> pd.DataFrame:
    rows = (
        tier_rows.groupby(
            [
                "claim_tier_rank",
                "claim_tier",
                "boundary_family_tier",
                "host_context_class_mode",
                "boundary_pattern_mode",
            ],
            as_index=False,
        )
        .agg(
            primitive_count=("primitive_id", "nunique"),
            event_count=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
            median_host_handle_mode_share=("dominant_host_handle_id_mode_share", "median"),
            median_shape_core_mode_share=("shape_core_signature_mode_share", "median"),
            median_fragmentation_index=("fragmentation_index_median", "median"),
        )
        .sort_values(["claim_tier_rank", "boundary_family_tier", "primitive_count"], ascending=[True, True, False])
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _gate_matrix(
    *,
    review_summary: dict[str, Any],
    tier_rows: pd.DataFrame,
    cumulative_ladder: pd.DataFrame,
) -> pd.DataFrame:
    route_status_ok = bool(
        tier_rows["route_execution_status"].eq(ROUTE_EXECUTION_STATUS).all()
        and tier_rows["wall_promotion_status"].eq(WALL_PROMOTION_STATUS).all()
        and tier_rows["quality_cost_status"].eq(QUALITY_COST_STATUS).all()
    )
    final = cumulative_ladder.sort_values("claim_tier_rank").tail(1).iloc[0]
    stable = cumulative_ladder[cumulative_ladder["claim_tier"].eq("T1_stable_high_support_nucleus")].iloc[0]
    rows = [
        {
            "gate_id": "L1_claim_ladder_accounting",
            "gate_question": "Does the tier ladder preserve the distribution-review accounting?",
            "evidence": (
                f"final_cumulative_primitives={int(final['cumulative_primitive_count'])}, "
                f"final_cumulative_events={int(final['cumulative_event_count'])}, "
                f"review_primitives={review_summary['accepted_primitive_count']}, "
                f"review_events={review_summary['accepted_event_count']}"
            ),
            "status": (
                "pass"
                if int(final["cumulative_primitive_count"]) == int(review_summary["accepted_primitive_count"])
                and int(final["cumulative_event_count"]) == int(review_summary["accepted_event_count"])
                else "blocked"
            ),
            "decision": "use_six_tier_ladder_for_result_wording",
            "next_action": "state cumulative scope explicitly whenever moving beyond T1",
        },
        {
            "gate_id": "L2_headline_scope",
            "gate_question": "Where should the first headline result start?",
            "evidence": (
                f"T1_primitives={int(stable['cumulative_primitive_count'])}, "
                f"T1_events={int(stable['cumulative_event_count'])}, "
                f"T1_families={int(stable['cumulative_source_family_count'])}"
            ),
            "status": "pass",
            "decision": "headline_claim_starts_at_T1_not_all_223",
            "next_action": "treat tiers T2-T6 as extensions or caveats",
        },
        {
            "gate_id": "L3_wall_quality_gate",
            "gate_question": "Can the tier ladder open wall/pathway or quality/cost claims?",
            "evidence": "only measurement and distribution-review rows are used",
            "status": "closed_excluded_by_design" if route_status_ok else "blocked_status_leak",
            "decision": "keep_route_wall_quality_cost_claims_closed",
            "next_action": "pathway design remains a separate future protocol",
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
    tier_summary: pd.DataFrame,
    cumulative_ladder: pd.DataFrame,
    host_boundary_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    text = [
        "# NanoClustering V2.2 Claim-Tier Ladder",
        "",
        f"- tier_count: `{len(TIER_ORDER)}`",
        f"- T1_stable_primitives: `{summary['tier_counts']['T1_stable_high_support_nucleus']}`",
        f"- T2_thin_clean_extension_primitives: `{summary['tier_counts']['T2_thin_clean_extension']}`",
        f"- T3_thin_concentration_caveat_primitives: `{summary['tier_counts']['T3_thin_concentration_caveat']}`",
        f"- T4_concentration_caveat_primitives: `{summary['tier_counts']['T4_concentration_caveat_no_residual']}`",
        f"- T5_standard_residual_primitives: `{summary['tier_counts']['T5_standard_residual_debt_caveat']}`",
        f"- T6_high_residual_priority_primitives: `{summary['tier_counts']['T6_high_residual_debt_priority']}`",
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
        "## Cumulative Ladder",
        "",
        _markdown_table(
            cumulative_ladder,
            [
                "claim_tier_rank",
                "claim_tier",
                "incremental_primitive_count",
                "incremental_event_count",
                "incremental_source_family_count",
                "cumulative_primitive_count",
                "cumulative_event_count",
                "cumulative_source_family_count",
                "headline_use",
            ],
            max_rows=10,
        ),
        "",
        "## Tier Summary",
        "",
        _markdown_table(
            tier_summary,
            [
                "claim_tier_rank",
                "claim_tier",
                "primitive_count",
                "event_count",
                "source_family_count",
                "persistent_mixed_core_primitives",
                "repeat_severe_core_primitives",
                "median_host_handle_mode_share",
                "median_shape_core_mode_share",
            ],
            max_rows=10,
        ),
        "",
        "## Host-Boundary Summary",
        "",
        _markdown_table(
            host_boundary_summary,
            [
                "claim_tier",
                "boundary_family_tier",
                "host_context_class_mode",
                "boundary_pattern_mode",
                "primitive_count",
                "event_count",
                "source_family_count",
            ],
            max_rows=35,
        ),
        "",
        "## Read",
        "",
        "- Use T1 as the headline evidence set: 83 stable high-support primitives, 452 events, and 79 source families.",
        "- T2 extends T1 to 106 primitives with thin-clean evidence; it is support-limited but not residual-debt or concentration-caveated.",
        "- T3 extends to 125 primitives by adding thin rows that also have concentration caveats.",
        "- T4 extends to 171 primitives by adding non-residual but concentration-caveated rows; these carry mechanism heterogeneity rather than clean support.",
        "- T5 extends to 218 primitives by adding standard residual-debt caveats; T6 reaches the full 223 by adding five high-residual-debt priority rows.",
        "- This ladder changes wording and ordering only. It does not change v2.2 membership or open wall/pathway, quality/cost, or directed-search claims.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    distribution_review_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    review_rows = _read_csv(distribution_review_dir / DISTRIBUTION_PRIMITIVE_ROWS_CSV)
    residual_priority_rows = _read_csv(distribution_review_dir / RESIDUAL_DEBT_PRIORITY_ROWS_CSV)
    review_summary = json.loads(
        (distribution_review_dir / DISTRIBUTION_REVIEW_SUMMARY_JSON).read_text(encoding="utf-8")
    )

    tier_rows = _claim_tier_rows(
        review_rows=review_rows,
        residual_priority_rows=residual_priority_rows,
    )
    tier_summary = _tier_summary(tier_rows)
    cumulative_ladder = _cumulative_ladder(tier_rows, tier_summary)
    host_boundary_summary = _host_boundary_summary(tier_rows)
    gate_matrix = _gate_matrix(
        review_summary=review_summary,
        tier_rows=tier_rows,
        cumulative_ladder=cumulative_ladder,
    )

    tier_counts = {
        tier: int(tier_rows.loc[tier_rows["claim_tier"].eq(tier), "primitive_id"].nunique())
        for tier in TIER_ORDER
    }
    tier_event_counts = {
        tier: int(tier_rows.loc[tier_rows["claim_tier"].eq(tier), "event_count"].sum())
        for tier in TIER_ORDER
    }
    summary = {
        "accepted_primitive_count": int(tier_rows["primitive_id"].nunique()),
        "accepted_event_count": int(tier_rows["event_count"].sum()),
        "accepted_source_family_count": int(tier_rows["source_family_id"].nunique()),
        "tier_counts": tier_counts,
        "tier_event_counts": tier_event_counts,
        "cumulative_ladder": cumulative_ladder[
            [
                "claim_tier",
                "cumulative_primitive_count",
                "cumulative_event_count",
                "cumulative_source_family_count",
            ]
        ].to_dict(orient="records"),
        "gate_status_counts": _count(gate_matrix, "status"),
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {"distribution_review_dir": _rel(distribution_review_dir)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(tier_rows, output_dir / CLAIM_TIER_PRIMITIVE_ROWS_CSV)
    _write_csv(tier_summary, output_dir / CLAIM_TIER_SUMMARY_CSV)
    _write_csv(cumulative_ladder, output_dir / CLAIM_TIER_CUMULATIVE_LADDER_CSV)
    _write_csv(host_boundary_summary, output_dir / CLAIM_TIER_HOST_BOUNDARY_SUMMARY_CSV)
    _write_csv(gate_matrix, output_dir / CLAIM_TIER_GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "distribution_review_dir": _rel(distribution_review_dir),
        "output_dir": _rel(output_dir),
        "tier_order": TIER_ORDER,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        tier_summary=tier_summary,
        cumulative_ladder=cumulative_ladder,
        host_boundary_summary=host_boundary_summary,
        gate_matrix=gate_matrix,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distribution-review-dir",
        type=Path,
        default=DEFAULT_DISTRIBUTION_REVIEW_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = materialize(
        distribution_review_dir=args.distribution_review_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
