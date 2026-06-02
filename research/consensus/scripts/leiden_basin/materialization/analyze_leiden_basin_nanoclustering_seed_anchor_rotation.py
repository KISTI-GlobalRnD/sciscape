#!/usr/bin/env python3
"""Audit whether NanoClustering endpoint-boundary signals survive seed-anchor rotation.

The current v2.2 primitive surface is anchored on seed0 reference clusters. This
audit rotates the reference seed across the pure Java/Rust seed ensembles and
checks whether recurrent fragmentation-like structure survives outside seed0.

This is still membership-only endpoint cartography. It does not run clustering,
execute routes, promote wall/pathway claims, inspect basin quality/cost, or
validate a basin-tunneling algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_LANDSCAPE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_external_landscape_20260530"
)
DEFAULT_CLAIM_TIER_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_v2_2_claim_tier_ladder_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_seed_anchor_rotation_audit_20260531"
)

ENDPOINT_REGISTRY_CSV = "nanoclustering_external_endpoint_registry.csv"
CLAIM_TIER_PRIMITIVE_ROWS_CSV = "nanoclustering_v2_2_claim_tier_primitive_rows.csv"

ROTATED_BEST_MATCH_ROWS_CSV = "nanoclustering_seed_anchor_rotation_best_match_rows.csv"
ROTATED_CLUSTER_SUMMARY_CSV = "nanoclustering_seed_anchor_rotation_cluster_summary.csv"
ROTATED_ANCHOR_SUMMARY_CSV = "nanoclustering_seed_anchor_rotation_anchor_summary.csv"
SEED0_TIER_RECOVERY_CSV = "nanoclustering_seed_anchor_rotation_seed0_tier_recovery.csv"
GATE_MATRIX_CSV = "nanoclustering_seed_anchor_rotation_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_seed_anchor_rotation_summary.json"
REPORT_MD = "nanoclustering_seed_anchor_rotation_report.md"
CONFIG_JSON = "nanoclustering_seed_anchor_rotation_config.json"

CLAIM_BOUNDARY = (
    "Seed-anchor rotation audit only; membership-derived endpoint cartography, "
    "no route execution, wall/pathway promotion, basin-quality claim, cost "
    "claim, directed-search claim, or algorithm claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_seed_anchor_rotation_audit"

SEVERE_TOP_SPLIT_THRESHOLD = 0.35
STRONG_TOP_SPLIT_THRESHOLD = 0.50
MODERATE_TOP_SPLIT_THRESHOLD = 0.80
RECURRENT_EVENT_MIN = 2
PERSISTENT_EVENT_MIN = 5

TIER_ORDER = [
    "T1_stable_high_support_nucleus",
    "T2_thin_clean_extension",
    "T3_thin_concentration_caveat",
    "T4_concentration_caveat_no_residual",
    "T5_standard_residual_debt_caveat",
    "T6_high_residual_debt_priority",
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


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _pure_seed_registry(landscape_dir: Path) -> pd.DataFrame:
    registry = _read_csv(landscape_dir / ENDPOINT_REGISTRY_CSV)
    pure = registry[registry["pure_seed_ensemble"].astype(str).eq("True")].copy()
    pure["seed"] = pure["seed"].astype(int)
    if pure.empty:
        raise ValueError("no pure seed endpoints in registry")
    return pure.sort_values(["branch", "seed"]).reset_index(drop=True)


def _load_seed_memberships(registry: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for row in registry.itertuples(index=False):
        label_cols = str(row.label_cols).split(";")
        if "candidate_micro_id" not in label_cols:
            raise ValueError(f"expected candidate_micro_id label: {row.run_id}")
        path = Path(row.absolute_path)
        table = pq.read_table(
            path,
            columns=[row.unit_col, row.weight_col, "candidate_micro_id"],
        )
        frame = table.to_pandas().rename(
            columns={
                row.unit_col: "unit_id",
                row.weight_col: "unit_weight",
                "candidate_micro_id": "cluster_id",
            }
        )
        frame["unit_id"] = frame["unit_id"].astype("int64")
        frame["unit_weight"] = frame["unit_weight"].astype("int64")
        frame["cluster_id"] = frame["cluster_id"].astype("int64")
        frames[(str(row.branch), int(row.seed))] = frame.sort_values(
            ["unit_id", "unit_weight"]
        ).reset_index(drop=True)
    return frames


def _event_band(top_split_share: float) -> str:
    if top_split_share < SEVERE_TOP_SPLIT_THRESHOLD:
        return "severe_fragmentation_event_lt_0p35"
    if top_split_share < STRONG_TOP_SPLIT_THRESHOLD:
        return "strong_fragmentation_event_lt_0p50"
    if top_split_share < MODERATE_TOP_SPLIT_THRESHOLD:
        return "moderate_fragmentation_event_lt_0p80"
    return "stable_retention_event_ge_0p80"


def _cluster_rule(row: pd.Series) -> str:
    strong_count = int(row["strong_fragmentation_event_count"])
    severe_count = int(row["severe_fragmentation_event_count"])
    moderate_count = int(row["moderate_fragmentation_event_count"])
    if strong_count >= PERSISTENT_EVENT_MIN:
        return "persistent_strong_fragmentation_candidate"
    if strong_count >= RECURRENT_EVENT_MIN:
        return "recurrent_strong_fragmentation_candidate"
    if severe_count >= 1:
        return "single_severe_fragmentation_candidate"
    if strong_count >= 1:
        return "single_strong_fragmentation_candidate"
    if moderate_count >= 1:
        return "moderate_fragmentation_candidate"
    return "matched_stable_like_reference"


def _best_match_rows_for_pair(
    *,
    branch: str,
    reference_seed: int,
    comparison_seed: int,
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    ref = reference[["unit_id", "unit_weight", "cluster_id"]].rename(
        columns={"cluster_id": "ref_cluster_id"}
    )
    other = comparison[["unit_id", "unit_weight", "cluster_id"]].rename(
        columns={"cluster_id": "run_cluster_id"}
    )
    ref_totals = ref.groupby("ref_cluster_id", as_index=False).agg(
        ref_unit_count=("unit_id", "size"),
        ref_weight_sum=("unit_weight", "sum"),
    )
    aligned = ref.merge(other, on=["unit_id", "unit_weight"], how="inner")
    overlap = aligned.groupby(["ref_cluster_id", "run_cluster_id"], as_index=False).agg(
        overlap_unit_count=("unit_id", "size"),
        overlap_weight_sum=("unit_weight", "sum"),
    )
    overlap = overlap.merge(ref_totals, on="ref_cluster_id", how="left")
    overlap["best_share_ref_units"] = overlap["overlap_unit_count"] / overlap["ref_unit_count"]
    overlap["best_share_ref_weight"] = overlap["overlap_weight_sum"] / overlap["ref_weight_sum"]
    overlap = overlap.sort_values(
        ["ref_cluster_id", "best_share_ref_weight", "best_share_ref_units", "run_cluster_id"],
        ascending=[True, False, False, True],
    )
    best = overlap.drop_duplicates("ref_cluster_id", keep="first").copy()
    best.insert(0, "branch", branch)
    best.insert(1, "reference_seed", reference_seed)
    best.insert(2, "comparison_seed", comparison_seed)
    best["top_split_share_ref_weight"] = best["best_share_ref_weight"].astype(float)
    best["fragmentation_index"] = 1.0 - best["top_split_share_ref_weight"]
    best["fragmentation_event_band"] = best["top_split_share_ref_weight"].map(_event_band)
    best["is_severe_fragmentation_event"] = best["top_split_share_ref_weight"].lt(
        SEVERE_TOP_SPLIT_THRESHOLD
    )
    best["is_strong_fragmentation_event"] = best["top_split_share_ref_weight"].lt(
        STRONG_TOP_SPLIT_THRESHOLD
    )
    best["is_moderate_fragmentation_event"] = best["top_split_share_ref_weight"].lt(
        MODERATE_TOP_SPLIT_THRESHOLD
    )
    return best


def _rotated_best_match_rows(
    registry: pd.DataFrame,
    memberships: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for branch, group in registry.groupby("branch", sort=True):
        seeds = sorted(int(seed) for seed in group["seed"].unique())
        for reference_seed in seeds:
            reference = memberships[(str(branch), reference_seed)]
            for comparison_seed in seeds:
                if comparison_seed == reference_seed:
                    continue
                rows.append(
                    _best_match_rows_for_pair(
                        branch=str(branch),
                        reference_seed=reference_seed,
                        comparison_seed=comparison_seed,
                        reference=reference,
                        comparison=memberships[(str(branch), comparison_seed)],
                    )
                )
    if not rows:
        return pd.DataFrame()
    best = pd.concat(rows, ignore_index=True, sort=False)
    preferred = [
        "branch",
        "reference_seed",
        "comparison_seed",
        "ref_cluster_id",
        "run_cluster_id",
        "ref_unit_count",
        "ref_weight_sum",
        "overlap_unit_count",
        "overlap_weight_sum",
        "best_share_ref_units",
        "best_share_ref_weight",
        "top_split_share_ref_weight",
        "fragmentation_index",
        "fragmentation_event_band",
        "is_severe_fragmentation_event",
        "is_strong_fragmentation_event",
        "is_moderate_fragmentation_event",
    ]
    return _with_claim_columns(best[preferred]).sort_values(
        ["branch", "reference_seed", "ref_cluster_id", "comparison_seed"]
    )


def _rotated_cluster_summary(best_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (branch, reference_seed, ref_cluster_id), group in best_rows.groupby(
        ["branch", "reference_seed", "ref_cluster_id"],
        sort=True,
    ):
        top_split = group["top_split_share_ref_weight"].astype(float)
        rows.append(
            {
                "branch": str(branch),
                "reference_seed": int(reference_seed),
                "ref_cluster_id": int(ref_cluster_id),
                "comparison_seed_count": int(group["comparison_seed"].nunique()),
                "ref_unit_count": int(group["ref_unit_count"].iloc[0]),
                "ref_weight_sum": int(group["ref_weight_sum"].iloc[0]),
                "top_split_share_min": float(top_split.min()),
                "top_split_share_q10": float(top_split.quantile(0.1)),
                "top_split_share_median": float(top_split.median()),
                "top_split_share_mean": float(top_split.mean()),
                "top_split_share_max": float(top_split.max()),
                "fragmentation_index_max": float(1.0 - top_split.min()),
                "fragmentation_index_median": float(1.0 - top_split.median()),
                "severe_fragmentation_event_count": int(
                    group["is_severe_fragmentation_event"].astype(bool).sum()
                ),
                "strong_fragmentation_event_count": int(
                    group["is_strong_fragmentation_event"].astype(bool).sum()
                ),
                "moderate_fragmentation_event_count": int(
                    group["is_moderate_fragmentation_event"].astype(bool).sum()
                ),
                "stable_retention_event_count": int(
                    group["top_split_share_ref_weight"].ge(MODERATE_TOP_SPLIT_THRESHOLD).sum()
                ),
                "most_fragmented_comparison_seed": int(
                    group.sort_values(
                        ["top_split_share_ref_weight", "comparison_seed"],
                        ascending=[True, True],
                    )["comparison_seed"].iloc[0]
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["anchor_fragmentation_rule"] = summary.apply(_cluster_rule, axis=1)
    summary["is_recurrent_strong_fragmentation_candidate"] = summary[
        "strong_fragmentation_event_count"
    ].ge(RECURRENT_EVENT_MIN)
    summary["is_persistent_strong_fragmentation_candidate"] = summary[
        "strong_fragmentation_event_count"
    ].ge(PERSISTENT_EVENT_MIN)
    summary["is_stable_like_reference"] = summary["moderate_fragmentation_event_count"].eq(0)
    return _with_claim_columns(summary).sort_values(
        ["branch", "reference_seed", "top_split_share_min", "ref_weight_sum"],
        ascending=[True, True, True, False],
    )


def _rotated_anchor_summary(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    rows = (
        cluster_summary.groupby(["branch", "reference_seed"], as_index=False)
        .agg(
            ref_cluster_count=("ref_cluster_id", "nunique"),
            ref_weight_sum=("ref_weight_sum", "sum"),
            top_split_share_min=("top_split_share_min", "min"),
            top_split_share_q10=("top_split_share_min", lambda s: float(s.quantile(0.1))),
            top_split_share_median=("top_split_share_median", "median"),
            recurrent_strong_fragmentation_candidate_count=(
                "is_recurrent_strong_fragmentation_candidate",
                "sum",
            ),
            persistent_strong_fragmentation_candidate_count=(
                "is_persistent_strong_fragmentation_candidate",
                "sum",
            ),
            stable_like_reference_count=("is_stable_like_reference", "sum"),
            severe_event_count=("severe_fragmentation_event_count", "sum"),
            strong_event_count=("strong_fragmentation_event_count", "sum"),
            moderate_event_count=("moderate_fragmentation_event_count", "sum"),
        )
        .sort_values(["branch", "reference_seed"])
    )
    numeric_int_cols = [
        "recurrent_strong_fragmentation_candidate_count",
        "persistent_strong_fragmentation_candidate_count",
        "stable_like_reference_count",
        "severe_event_count",
        "strong_event_count",
        "moderate_event_count",
    ]
    for column in numeric_int_cols:
        rows[column] = rows[column].astype(int)
    rows["recurrent_candidate_share"] = (
        rows["recurrent_strong_fragmentation_candidate_count"] / rows["ref_cluster_count"]
    )
    rows["persistent_candidate_share"] = (
        rows["persistent_strong_fragmentation_candidate_count"] / rows["ref_cluster_count"]
    )
    rows["stable_like_share"] = rows["stable_like_reference_count"] / rows["ref_cluster_count"]
    return _with_claim_columns(rows)


def _seed0_family_tiers(claim_tier_rows: pd.DataFrame) -> pd.DataFrame:
    rows = claim_tier_rows.copy()
    rows["source_family_id"] = rows["source_family_id"].astype(str)
    parsed = rows["source_family_id"].str.extract(
        r"^(?P<parsed_branch>[^_]+)_seed0_ref(?P<seed0_ref_cluster_id>\d+)$"
    )
    rows = pd.concat([rows, parsed], axis=1)
    rows = rows.dropna(subset=["parsed_branch", "seed0_ref_cluster_id"]).copy()
    if "branch" not in rows.columns:
        rows["branch"] = rows["parsed_branch"]
    elif not rows["branch"].astype(str).eq(rows["parsed_branch"].astype(str)).all():
        raise ValueError("source_family_id branch disagrees with claim-tier branch column")
    rows = rows.drop(columns=["parsed_branch"])
    rows["seed0_ref_cluster_id"] = rows["seed0_ref_cluster_id"].astype(int)
    rows["claim_tier_rank"] = rows["claim_tier"].map(
        {tier: index + 1 for index, tier in enumerate(TIER_ORDER)}
    )
    tier_lists = (
        rows.groupby(["branch", "source_family_id", "seed0_ref_cluster_id"], as_index=False)
        .agg(
            best_claim_tier_rank=("claim_tier_rank", "min"),
            claim_tiers=("claim_tier", lambda s: ";".join(sorted(set(map(str, s))))),
            primitive_count=("primitive_id", "nunique"),
            event_count=("event_count", "sum"),
        )
    )
    tier_lists["best_claim_tier"] = tier_lists["best_claim_tier_rank"].map(
        {index + 1: tier for index, tier in enumerate(TIER_ORDER)}
    )
    return tier_lists


def _seed0_mapping_from_anchor(
    memberships: dict[tuple[str, int], pd.DataFrame],
    *,
    branch: str,
    anchor_seed: int,
) -> pd.DataFrame:
    seed0 = memberships[(branch, 0)][["unit_id", "unit_weight", "cluster_id"]].rename(
        columns={"cluster_id": "seed0_ref_cluster_id"}
    )
    anchor = memberships[(branch, anchor_seed)][["unit_id", "unit_weight", "cluster_id"]].rename(
        columns={"cluster_id": "anchor_ref_cluster_id"}
    )
    anchor_totals = anchor.groupby("anchor_ref_cluster_id", as_index=False).agg(
        anchor_unit_count=("unit_id", "size"),
        anchor_weight_sum=("unit_weight", "sum"),
    )
    aligned = anchor.merge(seed0, on=["unit_id", "unit_weight"], how="inner")
    overlap = aligned.groupby(["anchor_ref_cluster_id", "seed0_ref_cluster_id"], as_index=False).agg(
        overlap_unit_count=("unit_id", "size"),
        overlap_weight_sum=("unit_weight", "sum"),
    )
    overlap = overlap.merge(anchor_totals, on="anchor_ref_cluster_id", how="left")
    overlap["seed0_share_of_anchor_weight"] = (
        overlap["overlap_weight_sum"] / overlap["anchor_weight_sum"]
    )
    overlap = overlap.sort_values(
        [
            "anchor_ref_cluster_id",
            "seed0_share_of_anchor_weight",
            "overlap_unit_count",
            "seed0_ref_cluster_id",
        ],
        ascending=[True, False, False, True],
    )
    best = overlap.drop_duplicates("anchor_ref_cluster_id", keep="first").copy()
    best.insert(0, "branch", branch)
    best.insert(1, "anchor_seed", anchor_seed)
    return best


def _seed0_tier_recovery(
    *,
    claim_tier_rows: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    memberships: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    seed0_families = _seed0_family_tiers(claim_tier_rows)
    rows: list[dict[str, Any]] = []
    for branch in sorted(seed0_families["branch"].unique()):
        branch = str(branch)
        for anchor_seed in sorted(
            seed
            for candidate_branch, seed in memberships
            if candidate_branch == branch and seed != 0
        ):
            mapping = _seed0_mapping_from_anchor(
                memberships,
                branch=branch,
                anchor_seed=int(anchor_seed),
            )
            anchor_clusters = cluster_summary[
                cluster_summary["branch"].eq(branch)
                & cluster_summary["reference_seed"].eq(int(anchor_seed))
            ][
                [
                    "ref_cluster_id",
                    "anchor_fragmentation_rule",
                    "is_recurrent_strong_fragmentation_candidate",
                    "is_persistent_strong_fragmentation_candidate",
                    "top_split_share_min",
                    "strong_fragmentation_event_count",
                ]
            ].rename(columns={"ref_cluster_id": "anchor_ref_cluster_id"})
            mapped = mapping.merge(
                anchor_clusters,
                on="anchor_ref_cluster_id",
                how="inner",
                validate="one_to_one",
            )
            joined = seed0_families[seed0_families["branch"].eq(branch)].merge(
                mapped,
                on=["branch", "seed0_ref_cluster_id"],
                how="left",
            )
            joined["recovered_by_recurrent_anchor"] = joined[
                "is_recurrent_strong_fragmentation_candidate"
            ].fillna(False).astype(bool)
            joined["recovered_by_persistent_anchor"] = joined[
                "is_persistent_strong_fragmentation_candidate"
            ].fillna(False).astype(bool)
            for tier in TIER_ORDER:
                group = joined[joined["claim_tiers"].str.contains(tier, regex=False)].copy()
                if group.empty:
                    continue
                rows.append(
                    {
                        "branch": branch,
                        "anchor_seed": int(anchor_seed),
                        "source_claim_tier": tier,
                        "seed0_source_family_count": int(group["source_family_id"].nunique()),
                        "recovered_by_recurrent_anchor_count": int(
                            group.loc[
                                group["recovered_by_recurrent_anchor"], "source_family_id"
                            ].nunique()
                        ),
                        "recovered_by_persistent_anchor_count": int(
                            group.loc[
                                group["recovered_by_persistent_anchor"], "source_family_id"
                            ].nunique()
                        ),
                        "mapped_anchor_cluster_count": int(
                            group["anchor_ref_cluster_id"].dropna().nunique()
                        ),
                        "median_seed0_share_of_anchor_weight": float(
                            group["seed0_share_of_anchor_weight"].median()
                        )
                        if group["seed0_share_of_anchor_weight"].notna().any()
                        else None,
                    }
                )
    recovery = pd.DataFrame(rows)
    if recovery.empty:
        return recovery
    recovery["recovered_by_recurrent_anchor_share"] = (
        recovery["recovered_by_recurrent_anchor_count"] / recovery["seed0_source_family_count"]
    )
    recovery["recovered_by_persistent_anchor_share"] = (
        recovery["recovered_by_persistent_anchor_count"] / recovery["seed0_source_family_count"]
    )
    recovery["route_execution_status"] = ROUTE_EXECUTION_STATUS
    recovery["wall_promotion_status"] = WALL_PROMOTION_STATUS
    recovery["quality_cost_status"] = QUALITY_COST_STATUS
    recovery["claim_boundary"] = CLAIM_BOUNDARY
    return recovery.sort_values(["branch", "anchor_seed", "source_claim_tier"])


def _gate_matrix(anchor_summary: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    non_seed0 = anchor_summary[anchor_summary["reference_seed"].ne(0)]
    min_recurrent = int(non_seed0["recurrent_strong_fragmentation_candidate_count"].min())
    max_recurrent = int(non_seed0["recurrent_strong_fragmentation_candidate_count"].max())
    t1 = recovery[recovery["source_claim_tier"].eq("T1_stable_high_support_nucleus")]
    min_t1_recovery = (
        float(t1["recovered_by_recurrent_anchor_share"].min()) if not t1.empty else 0.0
    )
    rows = [
        {
            "gate_id": "R1_anchor_rotation_executed",
            "gate_question": "Were all non-seed0 anchors evaluated for Java and Rust?",
            "evidence": (
                f"non_seed0_anchor_rows={len(non_seed0)}, "
                f"branches={anchor_summary['branch'].nunique()}"
            ),
            "status": "pass" if len(non_seed0) == 18 else "blocked_incomplete_anchor_grid",
            "decision": "use_rotation_audit_as_next_generality_gate",
            "next_action": "inspect anchor-level and T1 recovery distributions",
        },
        {
            "gate_id": "R2_non_seed0_recurrent_structure",
            "gate_question": "Do non-seed0 anchors also expose recurrent fragmentation candidates?",
            "evidence": f"min_recurrent={min_recurrent}, max_recurrent={max_recurrent}",
            "status": "pass" if min_recurrent > 0 else "blocked_no_recurrent_non_seed0_anchor",
            "decision": "phenomenon_not_seed0_only_at_count_level",
            "next_action": "test family-level recovery and symmetric endpoint objects",
        },
        {
            "gate_id": "R3_seed0_t1_family_recovery",
            "gate_question": "Are seed0 T1 source families recovered by non-seed0 recurrent anchors?",
            "evidence": f"min_T1_recurrent_recovery_share={min_t1_recovery:.6f}",
            "status": "caveat_required" if min_t1_recovery < 0.5 else "pass",
            "decision": "do_not_claim_seed_invariant_taxonomy_from_rotation_counts_alone",
            "next_action": "build symmetric_endpoint_object_audit before wall/pathway work",
        },
        {
            "gate_id": "R4_route_quality_gate",
            "gate_question": "Can this audit open wall/pathway, quality/cost, or method claims?",
            "evidence": "membership overlap only",
            "status": "closed_excluded_by_design",
            "decision": "keep_wall_quality_method_claims_closed",
            "next_action": "use only as generality evidence",
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix["claim_boundary"] = CLAIM_BOUNDARY
    return matrix


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for _, row in rows.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", r"\|"))
        body.append("| " + " | ".join(values) + " |")
    suffix = [f"\n_Showing {max_rows} of {len(frame)} rows._"] if len(frame) > max_rows else []
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    anchor_summary: pd.DataFrame,
    recovery: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    t1 = recovery[recovery["source_claim_tier"].eq("T1_stable_high_support_nucleus")]
    text = [
        "# NanoClustering Seed-Anchor Rotation Audit",
        "",
        f"- rotated_best_match_rows: `{summary['rotated_best_match_row_count']}`",
        f"- rotated_cluster_summary_rows: `{summary['rotated_cluster_summary_row_count']}`",
        f"- anchor_summary_rows: `{summary['anchor_summary_row_count']}`",
        f"- non_seed0_anchor_count: `{summary['non_seed0_anchor_count']}`",
        f"- min_non_seed0_recurrent_candidates: `{summary['min_non_seed0_recurrent_candidates']}`",
        f"- max_non_seed0_recurrent_candidates: `{summary['max_non_seed0_recurrent_candidates']}`",
        f"- min_T1_recurrent_recovery_share: `{summary['min_t1_recurrent_recovery_share']}`",
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
        "## Anchor Summary",
        "",
        _markdown_table(
            anchor_summary,
            [
                "branch",
                "reference_seed",
                "ref_cluster_count",
                "recurrent_strong_fragmentation_candidate_count",
                "persistent_strong_fragmentation_candidate_count",
                "stable_like_reference_count",
                "top_split_share_q10",
                "top_split_share_median",
            ],
            max_rows=25,
        ),
        "",
        "## T1 Recovery By Non-Seed0 Anchor",
        "",
        _markdown_table(
            t1,
            [
                "branch",
                "anchor_seed",
                "source_claim_tier",
                "seed0_source_family_count",
                "recovered_by_recurrent_anchor_count",
                "recovered_by_recurrent_anchor_share",
                "recovered_by_persistent_anchor_count",
                "recovered_by_persistent_anchor_share",
            ],
            max_rows=25,
        ),
        "",
        "## Read",
        "",
        "- This audit rotates the reference seed, so it directly tests the largest weakness in the seed0-anchored v2.2 surface.",
        "- Passing R2 means the phenomenon is not seed0-only at the coarse count level.",
        "- R3 is intentionally stricter: it asks whether seed0 T1 source families are recovered by non-seed0 recurrent anchors.",
        "- This audit still does not build anchor-independent basin objects. The next gate remains `symmetric_endpoint_object_audit` before any wall/pathway or method claim.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    landscape_dir: Path,
    claim_tier_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    registry = _pure_seed_registry(landscape_dir)
    claim_tier_rows = _read_csv(claim_tier_dir / CLAIM_TIER_PRIMITIVE_ROWS_CSV)
    memberships = _load_seed_memberships(registry)

    best_rows = _rotated_best_match_rows(registry, memberships)
    cluster_summary = _rotated_cluster_summary(best_rows)
    anchor_summary = _rotated_anchor_summary(cluster_summary)
    recovery = _seed0_tier_recovery(
        claim_tier_rows=claim_tier_rows,
        cluster_summary=cluster_summary,
        memberships=memberships,
    )
    gate_matrix = _gate_matrix(anchor_summary, recovery)

    non_seed0 = anchor_summary[anchor_summary["reference_seed"].ne(0)]
    t1 = recovery[recovery["source_claim_tier"].eq("T1_stable_high_support_nucleus")]
    min_t1_recovery = (
        float(t1["recovered_by_recurrent_anchor_share"].min()) if not t1.empty else None
    )
    summary = {
        "rotated_best_match_row_count": int(len(best_rows)),
        "rotated_cluster_summary_row_count": int(len(cluster_summary)),
        "anchor_summary_row_count": int(len(anchor_summary)),
        "non_seed0_anchor_count": int(len(non_seed0)),
        "branch_count": int(anchor_summary["branch"].nunique()),
        "reference_seed_values": sorted(int(seed) for seed in anchor_summary["reference_seed"].unique()),
        "comparison_seed_count_per_anchor": 9,
        "min_non_seed0_recurrent_candidates": int(
            non_seed0["recurrent_strong_fragmentation_candidate_count"].min()
        ),
        "max_non_seed0_recurrent_candidates": int(
            non_seed0["recurrent_strong_fragmentation_candidate_count"].max()
        ),
        "median_non_seed0_recurrent_candidates": float(
            non_seed0["recurrent_strong_fragmentation_candidate_count"].median()
        ),
        "min_t1_recurrent_recovery_share": min_t1_recovery,
        "median_t1_recurrent_recovery_share": float(
            t1["recovered_by_recurrent_anchor_share"].median()
        )
        if not t1.empty
        else None,
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gate_matrix["status"].value_counts().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {
            "landscape_dir": _rel(landscape_dir),
            "claim_tier_dir": _rel(claim_tier_dir),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(best_rows, output_dir / ROTATED_BEST_MATCH_ROWS_CSV)
    _write_csv(cluster_summary, output_dir / ROTATED_CLUSTER_SUMMARY_CSV)
    _write_csv(anchor_summary, output_dir / ROTATED_ANCHOR_SUMMARY_CSV)
    _write_csv(recovery, output_dir / SEED0_TIER_RECOVERY_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "landscape_dir": _rel(landscape_dir),
        "claim_tier_dir": _rel(claim_tier_dir),
        "output_dir": _rel(output_dir),
        "thresholds": {
            "severe_top_split": SEVERE_TOP_SPLIT_THRESHOLD,
            "strong_top_split": STRONG_TOP_SPLIT_THRESHOLD,
            "moderate_top_split": MODERATE_TOP_SPLIT_THRESHOLD,
            "recurrent_event_min": RECURRENT_EVENT_MIN,
            "persistent_event_min": PERSISTENT_EVENT_MIN,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        anchor_summary=anchor_summary,
        recovery=recovery,
        gate_matrix=gate_matrix,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--claim-tier-dir", type=Path, default=DEFAULT_CLAIM_TIER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = materialize(
        landscape_dir=args.landscape_dir,
        claim_tier_dir=args.claim_tier_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
