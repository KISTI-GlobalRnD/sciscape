#!/usr/bin/env python3
"""Materialize a fragmentation-first NanoClustering boundary inventory.

This applies the strongest matched-control signal, top-split retention, to the
full seed0 reference-cluster persistence universe. It does not materialize all
split/merge memberships, run clustering, execute routes, promote wall/pathway
claims, or inspect basin quality/cost.
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
DEFAULT_LANDSCAPE_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_external_landscape_20260530"
DEFAULT_CONTROL_DELTA_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_matched_control_delta_analysis_20260530"
)
DEFAULT_MATCHED_CONTROL_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_matched_controls_20260530"
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_fragmentation_boundary_inventory_20260530"
)

PERSISTENCE_BY_SEED_CSV = "nanoclustering_external_reference_cluster_persistence_by_seed.csv"
PERSISTENCE_SUMMARY_CSV = "nanoclustering_external_reference_cluster_persistence_summary.csv"
PAIR_DELTA_ROWS_CSV = "nanoclustering_matched_pair_delta_rows.csv"
MATCH_ROWS_CSV = "nanoclustering_volatile_to_stable_match_rows.csv"

EVENT_ROWS_CSV = "nanoclustering_fragmentation_boundary_event_rows.csv"
CLUSTER_INVENTORY_CSV = "nanoclustering_fragmentation_boundary_cluster_inventory.csv"
RULE_SUMMARY_CSV = "nanoclustering_fragmentation_boundary_rule_summary.csv"
COHORT_OVERLAP_CSV = "nanoclustering_fragmentation_boundary_cohort_overlap.csv"
TOP_CANDIDATES_CSV = "nanoclustering_fragmentation_boundary_top_candidates.csv"
SUMMARY_JSON = "nanoclustering_fragmentation_boundary_inventory_summary.json"
REPORT_MD = "nanoclustering_fragmentation_boundary_inventory_report.md"
CONFIG_JSON = "nanoclustering_fragmentation_boundary_inventory_config.json"

CLAIM_BOUNDARY = (
    "Fragmentation-first endpoint-boundary inventory only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_fragmentation_boundary_inventory"

SEVERE_TOP_SPLIT_THRESHOLD = 0.35
STRONG_TOP_SPLIT_THRESHOLD = 0.50
MODERATE_TOP_SPLIT_THRESHOLD = 0.80
RECURRENT_EVENT_MIN = 2
PERSISTENT_EVENT_MIN = 5


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


def _event_band(top_split_share: float) -> str:
    if top_split_share < SEVERE_TOP_SPLIT_THRESHOLD:
        return "severe_fragmentation_event_lt_0p35"
    if top_split_share < STRONG_TOP_SPLIT_THRESHOLD:
        return "strong_fragmentation_event_lt_0p50"
    if top_split_share < MODERATE_TOP_SPLIT_THRESHOLD:
        return "moderate_fragmentation_event_lt_0p80"
    return "stable_retention_event_ge_0p80"


def _cluster_rule(row: pd.Series) -> str:
    if int(row["strong_fragmentation_event_count"]) >= PERSISTENT_EVENT_MIN:
        return "persistent_strong_fragmentation_candidate"
    if int(row["strong_fragmentation_event_count"]) >= RECURRENT_EVENT_MIN:
        return "recurrent_strong_fragmentation_candidate"
    if int(row["severe_fragmentation_event_count"]) >= 1:
        return "single_severe_fragmentation_candidate"
    if int(row["strong_fragmentation_event_count"]) >= 1:
        return "single_strong_fragmentation_candidate"
    if int(row["moderate_fragmentation_event_count"]) >= 1:
        return "moderate_fragmentation_candidate"
    return "matched_stable_like_reference"


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _event_rows(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = by_seed.copy()
    rows["top_split_share_ref_weight"] = rows["best_share_ref_weight"].astype(float)
    rows["fragmentation_index"] = 1.0 - rows["top_split_share_ref_weight"]
    rows["fragmentation_event_band"] = rows["top_split_share_ref_weight"].map(_event_band)
    rows["is_severe_fragmentation_event"] = rows["top_split_share_ref_weight"].lt(
        SEVERE_TOP_SPLIT_THRESHOLD
    )
    rows["is_strong_fragmentation_event"] = rows["top_split_share_ref_weight"].lt(
        STRONG_TOP_SPLIT_THRESHOLD
    )
    rows["is_moderate_fragmentation_event"] = rows["top_split_share_ref_weight"].lt(
        MODERATE_TOP_SPLIT_THRESHOLD
    )
    rows["event_rule_note"] = (
        "top-split retention event band calibrated from matched-control delta analysis; "
        "membership-only endpoint evidence"
    )
    return _with_claim_columns(rows)


def _flag_sum(group: pd.DataFrame, column: str) -> int:
    return int(group[column].astype(bool).sum())


def _cluster_inventory(summary: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    event_agg_rows: list[dict[str, Any]] = []
    for (branch, ref_cluster_id), group in events.groupby(["branch", "ref_cluster_id"], sort=True):
        top_split = group["top_split_share_ref_weight"].astype(float)
        event_agg_rows.append(
            {
                "branch": str(branch),
                "ref_cluster_id": int(ref_cluster_id),
                "event_count": int(len(group)),
                "severe_fragmentation_event_count": _flag_sum(
                    group, "is_severe_fragmentation_event"
                ),
                "strong_fragmentation_event_count": _flag_sum(
                    group, "is_strong_fragmentation_event"
                ),
                "moderate_fragmentation_event_count": _flag_sum(
                    group, "is_moderate_fragmentation_event"
                ),
                "stable_retention_event_count": int(
                    group["top_split_share_ref_weight"].ge(MODERATE_TOP_SPLIT_THRESHOLD).sum()
                ),
                "top_split_share_min": float(top_split.min()),
                "top_split_share_q10": float(top_split.quantile(0.1)),
                "top_split_share_median": float(top_split.median()),
                "top_split_share_mean": float(top_split.mean()),
                "top_split_share_max": float(top_split.max()),
                "fragmentation_index_max": float(1.0 - top_split.min()),
                "fragmentation_index_median": float(1.0 - top_split.median()),
                "most_fragmented_comparison_seed": int(
                    group.sort_values(
                        ["top_split_share_ref_weight", "comparison_seed"],
                        ascending=[True, True],
                    )["comparison_seed"].iloc[0]
                ),
            }
        )
    event_agg = pd.DataFrame(event_agg_rows)
    rows = summary.merge(
        event_agg,
        on=["branch", "ref_cluster_id"],
        how="left",
        validate="one_to_one",
    )
    if rows.isna().any().any():
        missing = rows[rows.isna().any(axis=1)]
        raise ValueError(
            "summary/event aggregate mismatch: "
            f"{missing[['branch', 'ref_cluster_id']].to_dict('records')[:5]}"
        )
    rows["fragmentation_boundary_rule_v0"] = rows.apply(_cluster_rule, axis=1)
    rows["is_any_strong_fragmentation_candidate"] = rows["strong_fragmentation_event_count"].gt(0)
    rows["is_recurrent_strong_fragmentation_candidate"] = rows[
        "strong_fragmentation_event_count"
    ].ge(RECURRENT_EVENT_MIN)
    rows["is_persistent_strong_fragmentation_candidate"] = rows[
        "strong_fragmentation_event_count"
    ].ge(PERSISTENT_EVENT_MIN)
    rows["is_matched_stable_like_reference"] = rows["moderate_fragmentation_event_count"].eq(0)
    rows["boundary_rule_note"] = (
        "cluster-level primitive rule over seed0 reference persistence; "
        "fragmentation only, absorption and wall/pathway deferred"
    )
    rows = _with_claim_columns(rows)
    preferred = [
        "comparability_group",
        "branch",
        "ref_cluster_id",
        "ref_unit_count",
        "ref_weight_sum",
        "event_count",
        "top_split_share_min",
        "top_split_share_median",
        "fragmentation_index_max",
        "fragmentation_index_median",
        "severe_fragmentation_event_count",
        "strong_fragmentation_event_count",
        "moderate_fragmentation_event_count",
        "stable_retention_event_count",
        "most_fragmented_comparison_seed",
        "fragmentation_boundary_rule_v0",
        "is_any_strong_fragmentation_candidate",
        "is_recurrent_strong_fragmentation_candidate",
        "is_persistent_strong_fragmentation_candidate",
        "is_matched_stable_like_reference",
        "boundary_rule_note",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(
        [
            "branch",
            "is_persistent_strong_fragmentation_candidate",
            "is_recurrent_strong_fragmentation_candidate",
            "is_any_strong_fragmentation_candidate",
            "fragmentation_index_max",
            "ref_weight_sum",
        ],
        ascending=[True, False, False, False, False, False],
    )


def _rule_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    total_by_branch = inventory.groupby("branch").agg(
        branch_cluster_count=("ref_cluster_id", "size"),
        branch_ref_weight_sum=("ref_weight_sum", "sum"),
    )
    rows = []
    for (branch, rule), group in inventory.groupby(
        ["branch", "fragmentation_boundary_rule_v0"], sort=True
    ):
        branch_totals = total_by_branch.loc[branch]
        rows.append(
            {
                "branch": str(branch),
                "fragmentation_boundary_rule_v0": str(rule),
                "cluster_count": int(len(group)),
                "ref_weight_sum": int(group["ref_weight_sum"].sum()),
                "cluster_share_of_branch": float(
                    len(group) / branch_totals["branch_cluster_count"]
                ),
                "weight_share_of_branch": float(
                    group["ref_weight_sum"].sum() / branch_totals["branch_ref_weight_sum"]
                ),
                "median_top_split_share_min": float(group["top_split_share_min"].median()),
                "median_top_split_share_median": float(group["top_split_share_median"].median()),
                "median_strong_fragmentation_event_count": float(
                    group["strong_fragmentation_event_count"].median()
                ),
                "max_ref_weight_sum": int(group["ref_weight_sum"].max()),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["branch", "cluster_count"], ascending=[True, False]
    )


def _cohort_overlap(
    *,
    inventory: pd.DataFrame,
    pair_delta_rows: pd.DataFrame,
    match_rows: pd.DataFrame,
) -> pd.DataFrame:
    cohort_rows = []
    for row in match_rows.itertuples(index=False):
        cohort_rows.append(
            {
                "branch": row.branch,
                "ref_cluster_id": int(row.volatile_ref_cluster_id),
                "cohort": "volatile_selected",
                "matched_ref_cluster_id": int(row.control_ref_cluster_id),
            }
        )
        cohort_rows.append(
            {
                "branch": row.branch,
                "ref_cluster_id": int(row.control_ref_cluster_id),
                "cohort": "stable_matched_control",
                "matched_ref_cluster_id": int(row.volatile_ref_cluster_id),
            }
        )
    cohorts = pd.DataFrame(cohort_rows)
    rows = cohorts.merge(
        inventory,
        on=["branch", "ref_cluster_id"],
        how="left",
        validate="many_to_one",
    )
    if rows.isna().any().any():
        missing = rows[rows.isna().any(axis=1)]
        raise ValueError(
            f"cohort rows missing inventory: {missing[['branch', 'ref_cluster_id']].to_dict('records')}"
        )
    pair_axis = pair_delta_rows[
        [
            "branch",
            "volatile_ref_cluster_id",
            "control_ref_cluster_id",
            "pair_boundary_axis",
            "delta_fragmentation_index_median_volatile_minus_control",
        ]
    ]
    volatile_axis = pair_axis.rename(
        columns={
            "volatile_ref_cluster_id": "ref_cluster_id",
            "control_ref_cluster_id": "matched_ref_cluster_id",
        }
    )
    control_axis = pair_axis.rename(
        columns={
            "control_ref_cluster_id": "ref_cluster_id",
            "volatile_ref_cluster_id": "matched_ref_cluster_id",
        }
    )
    axis_rows = pd.concat([volatile_axis, control_axis], ignore_index=True, sort=False)
    rows = rows.merge(
        axis_rows,
        on=["branch", "ref_cluster_id", "matched_ref_cluster_id"],
        how="left",
        validate="one_to_one",
    )
    rows = _with_claim_columns(rows)
    columns = [
        "cohort",
        "branch",
        "ref_cluster_id",
        "matched_ref_cluster_id",
        "ref_unit_count",
        "ref_weight_sum",
        "top_split_share_min",
        "top_split_share_median",
        "severe_fragmentation_event_count",
        "strong_fragmentation_event_count",
        "moderate_fragmentation_event_count",
        "fragmentation_boundary_rule_v0",
        "pair_boundary_axis",
        "delta_fragmentation_index_median_volatile_minus_control",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    return rows[columns].sort_values(["cohort", "branch", "ref_cluster_id"])


def _top_candidates(inventory: pd.DataFrame, limit_per_branch: int) -> pd.DataFrame:
    candidates = inventory[inventory["is_any_strong_fragmentation_candidate"]].copy()
    ranked = candidates.sort_values(
        [
            "branch",
            "strong_fragmentation_event_count",
            "severe_fragmentation_event_count",
            "fragmentation_index_max",
            "ref_weight_sum",
        ],
        ascending=[True, False, False, False, False],
    )
    rows = ranked.groupby("branch", group_keys=False).head(limit_per_branch)
    columns = [
        "branch",
        "ref_cluster_id",
        "ref_unit_count",
        "ref_weight_sum",
        "top_split_share_min",
        "top_split_share_median",
        "fragmentation_index_max",
        "strong_fragmentation_event_count",
        "severe_fragmentation_event_count",
        "moderate_fragmentation_event_count",
        "most_fragmented_comparison_seed",
        "fragmentation_boundary_rule_v0",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    return rows[columns].reset_index(drop=True)


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
    inventory: pd.DataFrame,
    rule_summary: pd.DataFrame,
    cohort_overlap: pd.DataFrame,
    top_candidates: pd.DataFrame,
) -> None:
    total_clusters = len(inventory)
    any_strong = int(inventory["is_any_strong_fragmentation_candidate"].sum())
    recurrent = int(inventory["is_recurrent_strong_fragmentation_candidate"].sum())
    stable_like = int(inventory["is_matched_stable_like_reference"].sum())
    text = [
        "# NanoClustering Fragmentation Boundary Inventory",
        "",
        f"- cluster_universe: `{total_clusters}` seed0 reference clusters",
        f"- any_strong_fragmentation_candidate: `{any_strong}`",
        f"- recurrent_strong_fragmentation_candidate: `{recurrent}`",
        f"- matched_stable_like_reference: `{stable_like}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Rule Summary",
        "",
        _markdown_table(
            rule_summary,
            [
                "branch",
                "fragmentation_boundary_rule_v0",
                "cluster_count",
                "ref_weight_sum",
                "cluster_share_of_branch",
                "weight_share_of_branch",
                "median_top_split_share_min",
                "median_strong_fragmentation_event_count",
                "max_ref_weight_sum",
            ],
            max_rows=20,
        ),
        "",
        "## Volatile/Control Cohort Check",
        "",
        _markdown_table(
            cohort_overlap.groupby(["cohort", "fragmentation_boundary_rule_v0"], as_index=False)
            .agg(
                row_count=("ref_cluster_id", "size"),
                median_top_split_share_min=("top_split_share_min", "median"),
                median_strong_fragmentation_event_count=(
                    "strong_fragmentation_event_count",
                    "median",
                ),
            )
            .sort_values(["cohort", "row_count"], ascending=[True, False]),
            [
                "cohort",
                "fragmentation_boundary_rule_v0",
                "row_count",
                "median_top_split_share_min",
                "median_strong_fragmentation_event_count",
            ],
            max_rows=20,
        ),
        "",
        "## Top Candidates",
        "",
        _markdown_table(
            top_candidates,
            [
                "branch",
                "ref_cluster_id",
                "ref_weight_sum",
                "top_split_share_min",
                "top_split_share_median",
                "strong_fragmentation_event_count",
                "severe_fragmentation_event_count",
                "fragmentation_boundary_rule_v0",
            ],
            max_rows=30,
        ),
        "",
        "## Read",
        "",
        "- The 24 volatile case packets are not an isolated artifact: the same top-split fragmentation axis identifies a larger global inventory.",
        "- The rule family is intentionally primitive. It separates event-level strong fragmentation, recurrent strong fragmentation, and stable-like references without defining final global basins.",
        "- Absorption remains outside this inventory because matched controls showed absorption without fragmentation; absorption should be added later as an archetype axis only after event-level split/merge expansion.",
        "- This is endpoint cartography only. It does not establish optimizer-native walls, pathways, or quality/cost claims.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    landscape_dir: Path,
    control_delta_dir: Path,
    matched_control_dir: Path,
    output_dir: Path,
    top_limit_per_branch: int,
) -> dict[str, Any]:
    by_seed = _read_csv(landscape_dir / PERSISTENCE_BY_SEED_CSV)
    summary = _read_csv(landscape_dir / PERSISTENCE_SUMMARY_CSV)
    pair_delta_rows = _read_csv(control_delta_dir / PAIR_DELTA_ROWS_CSV)
    match_rows = _read_csv(matched_control_dir / MATCH_ROWS_CSV)

    events = _event_rows(by_seed)
    inventory = _cluster_inventory(summary, events)
    rule_summary = _rule_summary(inventory)
    cohort_overlap = _cohort_overlap(
        inventory=inventory,
        pair_delta_rows=pair_delta_rows,
        match_rows=match_rows,
    )
    top_candidates = _top_candidates(inventory, top_limit_per_branch)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(events, output_dir / EVENT_ROWS_CSV)
    _write_csv(inventory, output_dir / CLUSTER_INVENTORY_CSV)
    _write_csv(rule_summary, output_dir / RULE_SUMMARY_CSV)
    _write_csv(cohort_overlap, output_dir / COHORT_OVERLAP_CSV)
    _write_csv(top_candidates, output_dir / TOP_CANDIDATES_CSV)

    summary_payload = {
        "ok": True,
        "landscape_dir": _rel(landscape_dir),
        "control_delta_dir": _rel(control_delta_dir),
        "matched_control_dir": _rel(matched_control_dir),
        "output_dir": _rel(output_dir),
        "event_row_count": int(len(events)),
        "cluster_inventory_row_count": int(len(inventory)),
        "rule_summary_row_count": int(len(rule_summary)),
        "cohort_overlap_row_count": int(len(cohort_overlap)),
        "top_candidate_row_count": int(len(top_candidates)),
        "thresholds": {
            "severe_top_split_lt": SEVERE_TOP_SPLIT_THRESHOLD,
            "strong_top_split_lt": STRONG_TOP_SPLIT_THRESHOLD,
            "moderate_top_split_lt": MODERATE_TOP_SPLIT_THRESHOLD,
            "recurrent_event_min": RECURRENT_EVENT_MIN,
            "persistent_event_min": PERSISTENT_EVENT_MIN,
        },
        "global_counts": {
            "any_strong_fragmentation_candidate": int(
                inventory["is_any_strong_fragmentation_candidate"].sum()
            ),
            "recurrent_strong_fragmentation_candidate": int(
                inventory["is_recurrent_strong_fragmentation_candidate"].sum()
            ),
            "persistent_strong_fragmentation_candidate": int(
                inventory["is_persistent_strong_fragmentation_candidate"].sum()
            ),
            "matched_stable_like_reference": int(
                inventory["is_matched_stable_like_reference"].sum()
            ),
        },
        "rule_counts": {
            str(k): int(v)
            for k, v in inventory["fragmentation_boundary_rule_v0"].value_counts().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "quality_cost_status": QUALITY_COST_STATUS,
        "outputs": {
            "event_rows_csv": _rel(output_dir / EVENT_ROWS_CSV),
            "cluster_inventory_csv": _rel(output_dir / CLUSTER_INVENTORY_CSV),
            "rule_summary_csv": _rel(output_dir / RULE_SUMMARY_CSV),
            "cohort_overlap_csv": _rel(output_dir / COHORT_OVERLAP_CSV),
            "top_candidates_csv": _rel(output_dir / TOP_CANDIDATES_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "landscape_dir": str(landscape_dir),
        "control_delta_dir": str(control_delta_dir),
        "matched_control_dir": str(matched_control_dir),
        "output_dir": str(output_dir),
        "top_limit_per_branch": top_limit_per_branch,
        "thresholds": summary_payload["thresholds"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary_payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        inventory=inventory,
        rule_summary=rule_summary,
        cohort_overlap=cohort_overlap,
        top_candidates=top_candidates,
    )
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--control-delta-dir", type=Path, default=DEFAULT_CONTROL_DELTA_DIR)
    parser.add_argument("--matched-control-dir", type=Path, default=DEFAULT_MATCHED_CONTROL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-limit-per-branch", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        landscape_dir=args.landscape_dir.resolve(),
        control_delta_dir=args.control_delta_dir.resolve(),
        matched_control_dir=args.matched_control_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        top_limit_per_branch=args.top_limit_per_branch,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
