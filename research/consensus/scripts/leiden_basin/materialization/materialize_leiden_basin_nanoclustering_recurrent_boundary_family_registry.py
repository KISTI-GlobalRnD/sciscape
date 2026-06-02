#!/usr/bin/env python3
"""Materialize recurrent NanoClustering boundary-family registry.

This promotes the fragmentation inventory into a family-level registry for
recurrent strong endpoint-boundary candidates. It does not run clustering,
execute optimizer routes, promote wall/pathway claims, or inspect basin
quality/cost.
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
DEFAULT_INVENTORY_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_fragmentation_boundary_inventory_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_recurrent_boundary_family_registry_20260530"
)

INVENTORY_CSV = "nanoclustering_fragmentation_boundary_cluster_inventory.csv"
EVENT_ROWS_CSV = "nanoclustering_fragmentation_boundary_event_rows.csv"

FAMILY_REGISTRY_CSV = "nanoclustering_recurrent_boundary_family_registry.csv"
EVENT_SIGNATURE_ROWS_CSV = "nanoclustering_recurrent_boundary_event_signature_rows.csv"
FAMILY_TIER_SUMMARY_CSV = "nanoclustering_recurrent_boundary_family_tier_summary.csv"
PAIR_CONSTRUCTION_PANEL_CSV = "nanoclustering_recurrent_boundary_pair_construction_panel.csv"
SUMMARY_JSON = "nanoclustering_recurrent_boundary_family_registry_summary.json"
REPORT_MD = "nanoclustering_recurrent_boundary_family_registry_report.md"
CONFIG_JSON = "nanoclustering_recurrent_boundary_family_registry_config.json"

CLAIM_BOUNDARY = (
    "Recurrent endpoint-boundary family registry only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_recurrent_boundary_family_registry"

STRONG_TOP_SPLIT_THRESHOLD = 0.50
SEVERE_TOP_SPLIT_THRESHOLD = 0.35
RECURRENT_STRONG_MIN = 2
PERSISTENT_STRONG_MIN = 5


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


def _join_ints(values: pd.Series) -> str:
    return ",".join(f"{int(value):03d}" for value in sorted(set(values.dropna().astype(int))))


def _join_seed_targets(frame: pd.DataFrame) -> str:
    parts = []
    for row in frame.sort_values(["comparison_seed", "best_run_cluster_id"]).itertuples(index=False):
        parts.append(
            f"{int(row.comparison_seed):03d}:run{int(row.best_run_cluster_id)}:"
            f"top{float(row.top_split_share_ref_weight):.4f}"
        )
    return ";".join(parts)


def _family_tier(row: pd.Series) -> str:
    strong = int(row["strong_seed_count"])
    severe = int(row["severe_seed_count"])
    if severe >= 2:
        return "repeat_severe_core"
    if strong >= PERSISTENT_STRONG_MIN:
        return "persistent_mixed_core"
    if strong >= 3:
        return "multi_seed_mixed_recurrent"
    if severe == 1:
        return "pair_only_with_single_severe_edge"
    return "pair_only_nonsevere_edge"


def _readiness(row: pd.Series) -> str:
    tier = str(row["boundary_family_tier"])
    if tier in {"repeat_severe_core", "persistent_mixed_core"}:
        return "definition_core"
    if tier == "multi_seed_mixed_recurrent":
        return "definition_stress_test"
    return "edge_case_control"


def _expected_archetype(row: pd.Series) -> str:
    tier = str(row["boundary_family_tier"])
    if tier == "repeat_severe_core":
        return "severe_or_split_merge_likely"
    if tier == "persistent_mixed_core":
        return "repeated_fragmentation_mixed_archetype_expected"
    if tier == "multi_seed_mixed_recurrent":
        return "split_merge_likely_needs_expansion"
    if tier == "pair_only_with_single_severe_edge":
        return "seed_pair_discontinuity_or_boundary_edge"
    return "weak_recurrent_edge_needs_confirmation"


def _patch_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _event_signature_rows(inventory: pd.DataFrame, event_rows: pd.DataFrame) -> pd.DataFrame:
    recurrent = inventory[inventory["is_recurrent_strong_fragmentation_candidate"]].copy()
    key_cols = [
        "comparability_group",
        "branch",
        "ref_cluster_id",
        "fragmentation_boundary_rule_v0",
        "ref_unit_count",
        "ref_weight_sum",
        "top_split_share_min",
        "top_split_share_median",
        "strong_fragmentation_event_count",
        "severe_fragmentation_event_count",
        "moderate_fragmentation_event_count",
    ]
    rows = event_rows.merge(
        recurrent[key_cols],
        on=["comparability_group", "branch", "ref_cluster_id"],
        how="inner",
        suffixes=("_event", "_family"),
        validate="many_to_one",
    )
    rows["family_id"] = rows.apply(
        lambda row: f"{row['branch']}_seed0_ref{int(row['ref_cluster_id'])}",
        axis=1,
    )
    rows["top_split_share_ref_weight"] = rows["top_split_share_ref_weight"].astype(float)
    rows["is_strong_boundary_seed"] = rows["top_split_share_ref_weight"].lt(
        STRONG_TOP_SPLIT_THRESHOLD
    )
    rows["is_severe_boundary_seed"] = rows["top_split_share_ref_weight"].lt(
        SEVERE_TOP_SPLIT_THRESHOLD
    )
    rows["event_rank_by_fragmentation"] = (
        rows.sort_values(
            ["top_split_share_ref_weight", "comparison_seed"],
            ascending=[True, True],
        )
        .groupby(["branch", "ref_cluster_id"], sort=False)
        .cumcount()
        + 1
    )
    rows["event_signature_role"] = "stable_or_moderate_seed"
    rows.loc[rows["is_strong_boundary_seed"], "event_signature_role"] = "strong_boundary_seed"
    rows.loc[rows["is_severe_boundary_seed"], "event_signature_role"] = "severe_boundary_seed"
    rows["seed_target_key"] = rows.apply(
        lambda row: (
            f"{int(row['comparison_seed']):03d}:run{int(row['best_run_cluster_id'])}:"
            f"top{float(row['top_split_share_ref_weight']):.4f}"
        ),
        axis=1,
    )
    rows["event_registry_note"] = (
        "seed-level top-split signature for recurrent boundary-family registry; "
        "best_run_cluster_id is endpoint-local and not a cross-seed identity"
    )
    rows = _patch_claim_columns(rows)
    preferred = [
        "family_id",
        "comparability_group",
        "branch",
        "ref_cluster_id",
        "comparison_seed",
        "comparison_run_id",
        "best_run_cluster_id",
        "seed_target_key",
        "event_rank_by_fragmentation",
        "event_signature_role",
        "fragmentation_boundary_rule_v0",
        "ref_unit_count_family",
        "ref_weight_sum_family",
        "top_split_share_ref_weight",
        "fragmentation_index",
        "best_share_ref_units",
        "overlap_unit_count",
        "overlap_weight_sum",
        "is_strong_boundary_seed",
        "is_severe_boundary_seed",
        "event_registry_note",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(
        ["branch", "ref_cluster_id", "event_rank_by_fragmentation", "comparison_seed"]
    )


def _family_registry(event_signatures: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (branch, ref_cluster_id), group in event_signatures.groupby(
        ["branch", "ref_cluster_id"], sort=True
    ):
        strong = group[group["is_strong_boundary_seed"]]
        severe = group[group["is_severe_boundary_seed"]]
        moderate = group[group["top_split_share_ref_weight"].lt(0.80)]
        top = group["top_split_share_ref_weight"].astype(float)
        strong_top = strong["top_split_share_ref_weight"].astype(float)
        row = {
            "family_id": str(group["family_id"].iloc[0]),
            "comparability_group": str(group["comparability_group"].iloc[0]),
            "branch": str(branch),
            "ref_cluster_id": int(ref_cluster_id),
            "fragmentation_boundary_rule_v0": str(
                group["fragmentation_boundary_rule_v0"].iloc[0]
            ),
            "ref_unit_count": int(group["ref_unit_count_family"].iloc[0]),
            "ref_weight_sum": int(group["ref_weight_sum_family"].iloc[0]),
            "comparison_seed_count": int(group["comparison_seed"].nunique()),
            "strong_seed_count": int(strong["comparison_seed"].nunique()),
            "severe_seed_count": int(severe["comparison_seed"].nunique()),
            "moderate_seed_count": int(moderate["comparison_seed"].nunique()),
            "stable_or_mild_seed_count": int(group["top_split_share_ref_weight"].ge(0.80).sum()),
            "top_split_share_min": float(top.min()),
            "top_split_share_q10": float(top.quantile(0.1)),
            "top_split_share_median": float(top.median()),
            "top_split_share_mean": float(top.mean()),
            "top_split_share_max": float(top.max()),
            "strong_top_split_share_median": float(strong_top.median()),
            "strong_top_split_share_max": float(strong_top.max()),
            "fragmentation_index_max": float(1.0 - top.min()),
            "fragmentation_index_median": float(1.0 - top.median()),
            "strong_seed_list": _join_ints(strong["comparison_seed"]),
            "severe_seed_list": _join_ints(severe["comparison_seed"]),
            "moderate_seed_list": _join_ints(moderate["comparison_seed"]),
            "strong_seed_target_keys": _join_seed_targets(strong),
            "severe_seed_target_keys": _join_seed_targets(severe),
            "most_fragmented_comparison_seed": int(
                group.sort_values(
                    ["top_split_share_ref_weight", "comparison_seed"],
                    ascending=[True, True],
                )["comparison_seed"].iloc[0]
            ),
            "pair_construction_status": "needs_family_pair_construction",
            "family_registry_note": (
                "boundary family candidate from repeated top-split fragmentation; "
                "seed target keys are local handles for later split/merge pair construction"
            ),
        }
        rows.append(row)
    registry = pd.DataFrame(rows)
    registry["boundary_family_tier"] = registry.apply(_family_tier, axis=1)
    registry["definition_readiness"] = registry.apply(_readiness, axis=1)
    registry["expected_archetype_from_current_panel"] = registry.apply(_expected_archetype, axis=1)
    registry = _patch_claim_columns(registry)
    preferred = [
        "family_id",
        "comparability_group",
        "branch",
        "ref_cluster_id",
        "fragmentation_boundary_rule_v0",
        "boundary_family_tier",
        "definition_readiness",
        "expected_archetype_from_current_panel",
        "ref_unit_count",
        "ref_weight_sum",
        "comparison_seed_count",
        "strong_seed_count",
        "severe_seed_count",
        "moderate_seed_count",
        "stable_or_mild_seed_count",
        "top_split_share_min",
        "top_split_share_median",
        "strong_top_split_share_median",
        "fragmentation_index_max",
        "fragmentation_index_median",
        "strong_seed_list",
        "severe_seed_list",
        "strong_seed_target_keys",
        "severe_seed_target_keys",
        "most_fragmented_comparison_seed",
        "pair_construction_status",
        "family_registry_note",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in registry.columns if column not in preferred]
    return registry[preferred + remainder].sort_values(
        [
            "definition_readiness",
            "boundary_family_tier",
            "strong_seed_count",
            "severe_seed_count",
            "ref_weight_sum",
            "top_split_share_min",
        ],
        ascending=[True, True, False, False, False, True],
    )


def _family_tier_summary(registry: pd.DataFrame) -> pd.DataFrame:
    totals = registry.groupby("branch").agg(
        branch_family_count=("family_id", "size"),
        branch_ref_weight_sum=("ref_weight_sum", "sum"),
    )
    rows = []
    for (branch, tier), group in registry.groupby(["branch", "boundary_family_tier"], sort=True):
        branch_total = totals.loc[branch]
        rows.append(
            {
                "branch": str(branch),
                "boundary_family_tier": str(tier),
                "family_count": int(len(group)),
                "ref_weight_sum": int(group["ref_weight_sum"].sum()),
                "family_share_of_recurrent_branch": float(
                    len(group) / branch_total["branch_family_count"]
                ),
                "weight_share_of_recurrent_branch": float(
                    group["ref_weight_sum"].sum() / branch_total["branch_ref_weight_sum"]
                ),
                "median_ref_weight_sum": float(group["ref_weight_sum"].median()),
                "median_strong_seed_count": float(group["strong_seed_count"].median()),
                "median_severe_seed_count": float(group["severe_seed_count"].median()),
                "median_top_split_share_min": float(group["top_split_share_min"].median()),
                "median_top_split_share_median": float(group["top_split_share_median"].median()),
                "max_ref_weight_sum": int(group["ref_weight_sum"].max()),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    overall = []
    for tier, group in registry.groupby("boundary_family_tier", sort=True):
        overall.append(
            {
                "branch": "all",
                "boundary_family_tier": str(tier),
                "family_count": int(len(group)),
                "ref_weight_sum": int(group["ref_weight_sum"].sum()),
                "family_share_of_recurrent_branch": float(len(group) / len(registry)),
                "weight_share_of_recurrent_branch": float(
                    group["ref_weight_sum"].sum() / registry["ref_weight_sum"].sum()
                ),
                "median_ref_weight_sum": float(group["ref_weight_sum"].median()),
                "median_strong_seed_count": float(group["strong_seed_count"].median()),
                "median_severe_seed_count": float(group["severe_seed_count"].median()),
                "median_top_split_share_min": float(group["top_split_share_min"].median()),
                "median_top_split_share_median": float(group["top_split_share_median"].median()),
                "max_ref_weight_sum": int(group["ref_weight_sum"].max()),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame([*rows, *overall]).sort_values(
        ["branch", "family_count"], ascending=[True, False]
    )


def _pair_construction_panel(registry: pd.DataFrame, limit_per_branch_tier: int) -> pd.DataFrame:
    frames = []
    for (branch, tier), group in registry.groupby(["branch", "boundary_family_tier"], sort=True):
        if tier in {"repeat_severe_core", "persistent_mixed_core"}:
            sort_cols = [
                "severe_seed_count",
                "strong_seed_count",
                "ref_weight_sum",
                "fragmentation_index_max",
            ]
            ascending = [False, False, False, False]
        elif tier == "multi_seed_mixed_recurrent":
            sort_cols = ["strong_seed_count", "ref_weight_sum", "fragmentation_index_max"]
            ascending = [False, False, False]
        else:
            sort_cols = ["ref_weight_sum", "fragmentation_index_max", "strong_seed_count"]
            ascending = [False, False, False]
        selected = group.sort_values(sort_cols, ascending=ascending).head(limit_per_branch_tier).copy()
        selected["panel_rank_in_branch_tier"] = range(1, len(selected) + 1)
        selected["panel_selection_reason"] = (
            f"top {len(selected)} {tier} rows in {branch} for family-pair construction"
        )
        frames.append(selected)
    rows = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    rows = _patch_claim_columns(rows)
    preferred = [
        "panel_rank_in_branch_tier",
        "panel_selection_reason",
        "family_id",
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
        "strong_seed_target_keys",
        "pair_construction_status",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(
        ["branch", "boundary_family_tier", "panel_rank_in_branch_tier"]
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in rows.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
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
    registry: pd.DataFrame,
    event_signatures: pd.DataFrame,
    tier_summary: pd.DataFrame,
    panel: pd.DataFrame,
) -> None:
    core = registry[registry["definition_readiness"].eq("definition_core")]
    edge = registry[registry["definition_readiness"].eq("edge_case_control")]
    heavy_edges = edge.sort_values(["ref_weight_sum", "fragmentation_index_max"], ascending=[False, False])
    top_core = core.sort_values(
        ["strong_seed_count", "severe_seed_count", "ref_weight_sum"],
        ascending=[False, False, False],
    )
    text = [
        "# NanoClustering Recurrent Boundary-Family Registry",
        "",
        f"- recurrent_family_count: `{len(registry)}`",
        f"- event_signature_rows: `{len(event_signatures)}`",
        f"- definition_core_families: `{len(core)}`",
        f"- edge_case_control_families: `{len(edge)}`",
        f"- pair_construction_panel_rows: `{len(panel)}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Family Tier Summary",
        "",
        _markdown_table(
            tier_summary,
            [
                "branch",
                "boundary_family_tier",
                "family_count",
                "ref_weight_sum",
                "median_ref_weight_sum",
                "median_strong_seed_count",
                "median_severe_seed_count",
                "median_top_split_share_min",
                "max_ref_weight_sum",
            ],
            max_rows=24,
        ),
        "",
        "## Top Definition-Core Families",
        "",
        _markdown_table(
            top_core,
            [
                "family_id",
                "boundary_family_tier",
                "branch",
                "ref_cluster_id",
                "ref_weight_sum",
                "strong_seed_count",
                "severe_seed_count",
                "top_split_share_min",
                "top_split_share_median",
                "strong_seed_list",
            ],
            max_rows=30,
        ),
        "",
        "## Heavy Edge Cases",
        "",
        _markdown_table(
            heavy_edges,
            [
                "family_id",
                "boundary_family_tier",
                "branch",
                "ref_cluster_id",
                "ref_weight_sum",
                "strong_seed_count",
                "severe_seed_count",
                "top_split_share_min",
                "top_split_share_median",
                "strong_seed_list",
            ],
            max_rows=20,
        ),
        "",
        "## Pair-Construction Panel",
        "",
        _markdown_table(
            panel,
            [
                "panel_rank_in_branch_tier",
                "family_id",
                "boundary_family_tier",
                "definition_readiness",
                "branch",
                "ref_cluster_id",
                "ref_weight_sum",
                "strong_seed_count",
                "severe_seed_count",
                "top_split_share_min",
                "strong_seed_target_keys",
            ],
            max_rows=40,
        ),
        "",
        "## Read",
        "",
        "- This registry turns recurrent strong fragmentation into explicit endpoint-boundary family candidates.",
        "- `repeat_severe_core` and `persistent_mixed_core` are the current definition-core strata for the next pair-construction pass.",
        "- `pair_only_*` rows are retained as edge controls, especially when they are heavy but only recur in two seeds.",
        "- Seed target keys are local handles for later split/merge pair construction; they are not cross-seed endpoint identities.",
        "- This remains endpoint cartography only and does not establish optimizer-native wall/pathway evidence.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    inventory_dir: Path,
    output_dir: Path,
    limit_per_branch_tier: int,
) -> dict[str, Any]:
    inventory = _read_csv(inventory_dir / INVENTORY_CSV)
    event_rows = _read_csv(inventory_dir / EVENT_ROWS_CSV)
    event_signatures = _event_signature_rows(inventory, event_rows)
    registry = _family_registry(event_signatures)
    tier_summary = _family_tier_summary(registry)
    panel = _pair_construction_panel(registry, limit_per_branch_tier)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(registry, output_dir / FAMILY_REGISTRY_CSV)
    _write_csv(event_signatures, output_dir / EVENT_SIGNATURE_ROWS_CSV)
    _write_csv(tier_summary, output_dir / FAMILY_TIER_SUMMARY_CSV)
    _write_csv(panel, output_dir / PAIR_CONSTRUCTION_PANEL_CSV)

    summary = {
        "ok": True,
        "inventory_dir": _rel(inventory_dir),
        "output_dir": _rel(output_dir),
        "family_registry_row_count": int(len(registry)),
        "event_signature_row_count": int(len(event_signatures)),
        "family_tier_summary_row_count": int(len(tier_summary)),
        "pair_construction_panel_row_count": int(len(panel)),
        "definition_readiness_counts": {
            str(k): int(v) for k, v in registry["definition_readiness"].value_counts().to_dict().items()
        },
        "boundary_family_tier_counts": {
            str(k): int(v) for k, v in registry["boundary_family_tier"].value_counts().to_dict().items()
        },
        "thresholds": {
            "strong_top_split_lt": STRONG_TOP_SPLIT_THRESHOLD,
            "severe_top_split_lt": SEVERE_TOP_SPLIT_THRESHOLD,
            "recurrent_strong_min": RECURRENT_STRONG_MIN,
            "persistent_strong_min": PERSISTENT_STRONG_MIN,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "quality_cost_status": QUALITY_COST_STATUS,
        "outputs": {
            "family_registry_csv": _rel(output_dir / FAMILY_REGISTRY_CSV),
            "event_signature_rows_csv": _rel(output_dir / EVENT_SIGNATURE_ROWS_CSV),
            "family_tier_summary_csv": _rel(output_dir / FAMILY_TIER_SUMMARY_CSV),
            "pair_construction_panel_csv": _rel(output_dir / PAIR_CONSTRUCTION_PANEL_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "inventory_dir": str(inventory_dir),
        "output_dir": str(output_dir),
        "limit_per_branch_tier": limit_per_branch_tier,
        "thresholds": summary["thresholds"],
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
    _write_report(
        output_dir=output_dir,
        registry=registry,
        event_signatures=event_signatures,
        tier_summary=tier_summary,
        panel=panel,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-per-branch-tier", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        inventory_dir=args.inventory_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        limit_per_branch_tier=args.limit_per_branch_tier,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
