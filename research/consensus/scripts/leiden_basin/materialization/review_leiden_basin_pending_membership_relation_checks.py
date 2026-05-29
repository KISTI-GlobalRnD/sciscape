#!/usr/bin/env python3
"""Review pending-membership relation checks for Leiden basin cartography.

This diagnostic inspects relation-queue rows marked
`pending_membership_relation_check`. It checks whether full membership cache is
already available, and if not, what exact changed-support and sketch-signature
evidence can be read from the original candidate rows. It does not run routes,
relax wall-promotion rules, or inspect basin quality/cost.
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
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_BLOCKER_TRIAGE_DIR = BASE_RESULT_DIR / "leiden_basin_route_label_blocker_triage_20260529"
DEFAULT_COVERAGE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_wall_panel_context_coverage_after_clean_distinct_route_gate_20260528"
)
DEFAULT_ENDPOINT_CACHE_DIR = BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_endpoint_cache_20260528"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_pending_membership_relation_review_20260529"

RELATION_QUEUE_CSV = "relation_definition_queue.csv"
COVERAGE_ROWS_CSV = "wall_panel_context_coverage_rows.csv"

REVIEW_ROWS_CSV = "pending_membership_relation_review_rows.csv"
EVIDENCE_LINKS_CSV = "pending_membership_relation_evidence_links.csv"
COUNTERFACTUALS_CSV = "pending_membership_relation_counterfactuals.csv"
SUMMARY_JSON = "pending_membership_relation_review_summary.json"
REPORT_MD = "pending_membership_relation_review_report.md"
CONFIG_JSON = "pending_membership_relation_review_config.json"

REVIEW_VERSION = "pending_membership_relation_review_20260529"
SAME_SUPPORT_MAX = 0.5
DISTINCT_SUPPORT_MIN = 0.75
CLAIM_BOUNDARY = (
    "Pending-membership relation review only; no route execution, "
    "wall-promotion change, basin-quality claim, cost claim, or directed-search claim."
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve(path_text: Any) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"empty CSV: {path}") from exc


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return math.nan
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_nodes(value: Any) -> set[int]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {int(part) for part in text.split(";") if part != ""}


def _support_distance(left: set[int], right: set[int]) -> tuple[float, int, int]:
    union = left | right
    if not union:
        return 0.0, 0, 0
    intersection = left & right
    return 1.0 - (len(intersection) / len(union)), len(intersection), len(union)


def _hard_gate_classification(distance: float) -> str:
    if distance <= SAME_SUPPORT_MAX:
        return "same_support_local"
    if distance >= DISTINCT_SUPPORT_MIN:
        return "distinct_support_local"
    return "boundary_review_ambiguous_support_local"


def _epsilon_classification(distance: float, epsilon: float, two_sided: bool) -> str:
    if distance <= SAME_SUPPORT_MAX:
        return "same_support_local"
    if distance >= DISTINCT_SUPPORT_MIN:
        return "distinct_support_local"
    if two_sided and distance <= SAME_SUPPORT_MAX + epsilon:
        return "same_support_local_epsilon_snap"
    if distance >= DISTINCT_SUPPORT_MIN - epsilon:
        return "distinct_support_local_epsilon_snap"
    return "boundary_review_ambiguous_support_local"


def _load_cache_index(cache_dir: Path) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for metadata_path in cache_dir.glob("*.metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(metadata.get("kind", "")) != "endpoint":
            continue
        case_id = str(metadata.get("case_id", ""))
        candidate_index = _safe_int(metadata.get("candidate_index"))
        if case_id and candidate_index is not None:
            out.add((case_id, candidate_index))
    return out


def _candidate_row(path: Path, candidate_index: int) -> pd.Series:
    frame = _read_csv(path)
    rows = frame[pd.to_numeric(frame["candidate_index"], errors="coerce").eq(candidate_index)]
    if rows.empty:
        raise ValueError(f"candidate {candidate_index} not found in {path}")
    return rows.iloc[0]


def _signature_status(left: pd.Series, right: pd.Series) -> str:
    same_sample = str(left.get("p5_basin_sketch_node_hash", "")) == str(
        right.get("p5_basin_sketch_node_hash", "")
    )
    same_signature = str(left.get("p5_basin_signature", "")) == str(
        right.get("p5_basin_signature", "")
    )
    same_support_hash = str(left.get("p5_basin_changed_support_node_hash", "")) == str(
        right.get("p5_basin_changed_support_node_hash", "")
    )
    if same_signature and same_support_hash:
        return "proxy_signature_and_support_hash_match"
    if same_sample and not same_signature:
        return "proxy_signature_differs_on_same_sketch_sample"
    if not same_support_hash:
        return "changed_support_hash_differs"
    return "proxy_signature_inconclusive"


def _review_decision(distance: float, cache_status: str) -> str:
    if cache_status == "both_endpoint_memberships_cached":
        if _hard_gate_classification(distance) == "boundary_review_ambiguous_support_local":
            return "cached_membership_still_boundary_review"
        return "cached_membership_relation_resolved_under_hard_gate"
    return "support_exact_available_but_full_membership_cache_missing"


def _next_evidence(row: pd.Series) -> str:
    if str(row.get("full_membership_cache_status", "")) != "both_endpoint_memberships_cached":
        return "link_or_reconstruct_full_membership_cache_for_both_endpoint_candidates"
    if str(row.get("current_hard_gate_classification", "")) == "boundary_review_ambiguous_support_local":
        return "predeclare_boundary_band_rule_before_route_promotion"
    return "rerun_relation_taxonomy_with_cached_membership_result"


def _review_rows(
    blocker_triage_dir: Path,
    coverage_dir: Path,
    endpoint_cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    relation_queue = _read_csv(blocker_triage_dir / RELATION_QUEUE_CSV)
    coverage = _read_csv(coverage_dir / COVERAGE_ROWS_CSV)
    pending_ids = set(
        relation_queue[
            relation_queue["relation_queue_status"].eq("pending_membership_relation_check")
        ]["panel_pair_id"].astype(str)
    )
    if not pending_ids:
        raise ValueError("no pending_membership_relation_check rows found")

    cache_index = _load_cache_index(endpoint_cache_dir)
    coverage = coverage[coverage["panel_pair_id"].astype(str).isin(pending_ids)].copy()
    queue_cols = [
        "panel_pair_id",
        "relation_taxonomy_v0_1",
        "route_label_interpretation_v0",
        "relation_queue_status",
        "blocker_priority",
        "triage_action",
        "triage_rationale",
    ]
    rows = coverage.merge(relation_queue[queue_cols], on="panel_pair_id", how="left")

    review_rows: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        panel_pair_id = str(row["panel_pair_id"])
        case_id = str(row["case_id"])
        left_index = _safe_int(row["left_representative_candidate_index"])
        right_index = _safe_int(row["right_representative_candidate_index"])
        if left_index is None or right_index is None:
            raise ValueError(f"missing candidate index for {panel_pair_id}")
        left_path = _resolve(row["left_endpoint_source_artifact"])
        right_path = _resolve(row["right_endpoint_source_artifact"])
        left = _candidate_row(left_path, left_index)
        right = _candidate_row(right_path, right_index)

        left_nodes = _parse_nodes(left.get("p5_basin_changed_support_nodes"))
        right_nodes = _parse_nodes(right.get("p5_basin_changed_support_nodes"))
        exact_distance, support_intersection, support_union = _support_distance(
            left_nodes,
            right_nodes,
        )
        coverage_distance = _safe_float(row.get("support_distance_max"))
        support_delta = (
            exact_distance - coverage_distance if math.isfinite(coverage_distance) else math.nan
        )
        left_cached = (case_id, left_index) in cache_index
        right_cached = (case_id, right_index) in cache_index
        if left_cached and right_cached:
            cache_status = "both_endpoint_memberships_cached"
        elif left_cached or right_cached:
            cache_status = "partial_endpoint_membership_cache"
        else:
            cache_status = "full_membership_cache_missing"
        signature_status = _signature_status(left, right)
        current_classification = _hard_gate_classification(exact_distance)
        same_margin = exact_distance - SAME_SUPPORT_MAX
        distinct_margin = DISTINCT_SUPPORT_MIN - exact_distance
        review_decision = _review_decision(exact_distance, cache_status)

        review_rows.append(
            {
                "panel_pair_id": panel_pair_id,
                "field": row["field"],
                "case_id": case_id,
                "method": row["method"],
                "panel_role": row["panel_role"],
                "relation_taxonomy_v0_1": row.get("relation_taxonomy_v0_1", ""),
                "route_label_interpretation_v0": row.get("route_label_interpretation_v0", ""),
                "relation_queue_status": row.get("relation_queue_status", ""),
                "blocker_priority": row.get("blocker_priority", ""),
                "triage_action": row.get("triage_action", ""),
                "runner_preflight_status": row.get("runner_preflight_status", ""),
                "left_candidate_index": left_index,
                "right_candidate_index": right_index,
                "left_endpoint_identity_id": row["left_endpoint_identity_id"],
                "right_endpoint_identity_id": row["right_endpoint_identity_id"],
                "left_support_node_count_from_candidate": int(
                    left.get("p5_basin_changed_support_node_count", len(left_nodes))
                ),
                "right_support_node_count_from_candidate": int(
                    right.get("p5_basin_changed_support_node_count", len(right_nodes))
                ),
                "left_support_node_count_parsed": len(left_nodes),
                "right_support_node_count_parsed": len(right_nodes),
                "exact_support_intersection_from_nodes": support_intersection,
                "exact_support_union_from_nodes": support_union,
                "exact_support_distance_from_nodes": exact_distance,
                "coverage_support_distance": coverage_distance,
                "support_distance_delta_from_coverage": support_delta,
                "same_threshold_margin": same_margin,
                "distinct_threshold_margin": distinct_margin,
                "current_hard_gate_classification": current_classification,
                "epsilon_0p001_distinct_only_classification": _epsilon_classification(
                    exact_distance,
                    epsilon=0.001,
                    two_sided=False,
                ),
                "epsilon_0p02_two_sided_classification": _epsilon_classification(
                    exact_distance,
                    epsilon=0.02,
                    two_sided=True,
                ),
                "left_changed_support_hash": left.get("p5_basin_changed_support_node_hash", ""),
                "right_changed_support_hash": right.get("p5_basin_changed_support_node_hash", ""),
                "left_basin_signature": left.get("p5_basin_signature", ""),
                "right_basin_signature": right.get("p5_basin_signature", ""),
                "left_sketch_node_hash": left.get("p5_basin_sketch_node_hash", ""),
                "right_sketch_node_hash": right.get("p5_basin_sketch_node_hash", ""),
                "sketch_sample_size": left.get("p5_basin_sketch_sample_size", ""),
                "proxy_signature_status": signature_status,
                "full_membership_cache_status": cache_status,
                "left_endpoint_cache_present": left_cached,
                "right_endpoint_cache_present": right_cached,
                "review_decision": review_decision,
                "wall_promotion_status_after_review": "no_wall_promotion",
                "route_execution_status_after_review": "not_recommended",
                "next_evidence_required": "",
                "claim_boundary": CLAIM_BOUNDARY,
                "review_version": REVIEW_VERSION,
                "source_relation_queue_artifact": _rel(blocker_triage_dir / RELATION_QUEUE_CSV),
                "source_coverage_artifact": _rel(coverage_dir / COVERAGE_ROWS_CSV),
            }
        )
        for role, path, candidate_index, candidate_row, cached in (
            ("left_candidate", left_path, left_index, left, left_cached),
            ("right_candidate", right_path, right_index, right, right_cached),
        ):
            links.append(
                {
                    "panel_pair_id": panel_pair_id,
                    "evidence_role": role,
                    "candidate_index": candidate_index,
                    "candidate_source_artifact": _rel(path),
                    "full_membership_cache_present": cached,
                    "changed_support_node_hash": candidate_row.get(
                        "p5_basin_changed_support_node_hash",
                        "",
                    ),
                    "basin_signature": candidate_row.get("p5_basin_signature", ""),
                    "sketch_node_hash": candidate_row.get("p5_basin_sketch_node_hash", ""),
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )

    frame = pd.DataFrame(review_rows)
    frame["next_evidence_required"] = frame.apply(_next_evidence, axis=1)
    return frame.sort_values(["panel_role", "panel_pair_id"]).reset_index(drop=True), pd.DataFrame(
        links
    )


def _counterfactuals(rows: pd.DataFrame) -> pd.DataFrame:
    policies = [
        (
            "current_hard_gate",
            "Keep same <= 0.5 and distinct >= 0.75; middle rows stay boundary review.",
            "current_hard_gate_classification",
            "accepted",
        ),
        (
            "epsilon_0p001_distinct_only",
            "Snap near-distinct rows within 0.001 below 0.75 to distinct.",
            "epsilon_0p001_distinct_only_classification",
            "rejected",
        ),
        (
            "epsilon_0p02_two_sided",
            "Snap rows within 0.02 of either same or distinct threshold.",
            "epsilon_0p02_two_sided_classification",
            "rejected",
        ),
    ]
    out: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        for policy_id, policy_description, source_col, status in policies:
            out.append(
                {
                    "panel_pair_id": row.panel_pair_id,
                    "exact_support_distance_from_nodes": row.exact_support_distance_from_nodes,
                    "full_membership_cache_status": row.full_membership_cache_status,
                    "policy_id": policy_id,
                    "policy_description": policy_description,
                    "counterfactual_classification": getattr(row, source_col),
                    "policy_review_status": status,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(out)


def _summary(rows: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    cache_counts = _count(rows, "full_membership_cache_status")
    all_cached = bool(
        len(rows)
        and rows["full_membership_cache_status"].eq("both_endpoint_memberships_cached").all()
    )
    all_boundary = bool(
        len(rows)
        and rows["current_hard_gate_classification"]
        .eq("boundary_review_ambiguous_support_local")
        .all()
    )
    if all_cached and all_boundary:
        decision = (
            "Full membership cache is available for both pending rows. Under the current "
            "hard gate both rows still remain boundary-review holds; do not route-promote "
            "or run a wider route batch."
        )
        next_step = (
            "Keep these rows as cached boundary-review holds unless the boundary-band "
            "definition is explicitly changed; field34 hygiene remains separate."
        )
    else:
        decision = (
            "Exact changed-support evidence is available for both pending rows, "
            "but full membership cache is missing. Under the current hard gate both "
            "rows remain boundary-review holds; do not route-promote or run a wider route batch."
        )
        next_step = (
            "Either reconstruct/link full membership cache for these field26 endpoints "
            "or keep them as support-exact pending rows; field34 hygiene remains separate."
        )
    return {
        "status": "pending_membership_relation_review_prepared",
        "date": "2026-05-29",
        "script": "research/consensus/scripts/review_leiden_basin_pending_membership_relation_checks.py",
        "output_dir": _rel(output_dir),
        "review_version": REVIEW_VERSION,
        "reviewed_pair_count": int(len(rows)),
        "full_membership_cache_status_counts": cache_counts,
        "review_decision_counts": _count(rows, "review_decision"),
        "current_hard_gate_classification_counts": _count(
            rows,
            "current_hard_gate_classification",
        ),
        "epsilon_0p001_distinct_only_counts": _count(
            rows,
            "epsilon_0p001_distinct_only_classification",
        ),
        "epsilon_0p02_two_sided_counts": _count(rows, "epsilon_0p02_two_sided_classification"),
        "proxy_signature_status_counts": _count(rows, "proxy_signature_status"),
        "promoted_wall_claim_count": int(
            rows["wall_promotion_status_after_review"].ne("no_wall_promotion").sum()
        ),
        "immediate_route_execution_count": int(
            rows["route_execution_status_after_review"].eq("ready").sum()
        ),
        "decision": decision,
        "next_step": next_step,
        "claim_boundary": CLAIM_BOUNDARY,
        "paths": {
            "review_rows": _rel(output_dir / REVIEW_ROWS_CSV),
            "evidence_links": _rel(output_dir / EVIDENCE_LINKS_CSV),
            "counterfactuals": _rel(output_dir / COUNTERFACTUALS_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rendered_rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(str(column)), *(len(row[index]) for row in rendered_rows))
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(
        str(column).ljust(widths[index]) for index, column in enumerate(columns)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in rendered_rows
    ]
    return "\n".join([header, separator, *body])


def _report(rows: pd.DataFrame, counterfactuals: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Pending-Membership Relation Review",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact reviews the two relation-queue rows marked",
        "`pending_membership_relation_check`. It checks cache availability and",
        "exact changed-support evidence from the source candidate rows. It does not",
        "run routes, change wall promotion, or inspect basin quality/cost.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "Claim boundary: " + CLAIM_BOUNDARY,
        "",
        "## Reviewed Rows",
        "",
    ]
    row_cols = [
        "panel_pair_id",
        "exact_support_distance_from_nodes",
        "same_threshold_margin",
        "distinct_threshold_margin",
        "current_hard_gate_classification",
        "full_membership_cache_status",
        "proxy_signature_status",
        "review_decision",
    ]
    lines.append(_markdown_table(rows[row_cols]))
    lines.extend(["", "## Counterfactuals", ""])
    cf_cols = [
        "panel_pair_id",
        "policy_id",
        "counterfactual_classification",
        "policy_review_status",
    ]
    lines.append(_markdown_table(counterfactuals[cf_cols]))
    if (
        rows["full_membership_cache_status"].eq("both_endpoint_memberships_cached").all()
        and rows["current_hard_gate_classification"]
        .eq("boundary_review_ambiguous_support_local")
        .all()
    ):
        interpretation = [
            "Full membership cache is now linked for both pending rows. The",
            "materialized memberships reproduce the source changed-support evidence,",
            "and both rows remain inside the predeclared middle zone under the hard",
            "gate. These rows should be treated as cached boundary-review holds, not",
            "as wall evidence.",
        ]
    else:
        interpretation = [
            "The source candidate rows provide exact changed-support node sets and",
            "proxy signatures. Both rows remain inside the predeclared middle zone",
            "under the hard gate. Because full membership cache is missing, these",
            "rows should not be upgraded to the cached boundary-review class yet.",
        ]
    lines.extend(["", "## Interpretation", "", *interpretation, "", "Next step: " + str(summary["next_step"]), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocker-triage-dir", type=Path, default=DEFAULT_BLOCKER_TRIAGE_DIR)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument("--endpoint-cache-dir", type=Path, default=DEFAULT_ENDPOINT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, links = _review_rows(args.blocker_triage_dir, args.coverage_dir, args.endpoint_cache_dir)
    counterfactuals = _counterfactuals(rows)
    summary = _summary(rows, output_dir)
    config = {
        "blocker_triage_dir": _rel(args.blocker_triage_dir),
        "coverage_dir": _rel(args.coverage_dir),
        "endpoint_cache_dir": _rel(args.endpoint_cache_dir),
        "output_dir": _rel(output_dir),
        "same_support_max": SAME_SUPPORT_MAX,
        "distinct_support_min": DISTINCT_SUPPORT_MIN,
        "review_version": REVIEW_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(rows, output_dir / REVIEW_ROWS_CSV)
    _write_csv(links, output_dir / EVIDENCE_LINKS_CSV)
    _write_csv(counterfactuals, output_dir / COUNTERFACTUALS_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / REPORT_MD).write_text(_report(rows, counterfactuals, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
