#!/usr/bin/env python3
"""Calibrate Phase 1 Leiden basin definitions before wall cartography.

This script consumes existing endpoint and pairwise signature artifacts. It
does not use quality, materiality, cost, ranking, or operator-success fields to
define basin identities or basin relations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import pandas as pd

BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_PHASE1_DIR = BASE_RESULT_DIR / "leiden_basin_phase1_index_20260528"
DEFAULT_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_phase1_review_20260528"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_definition_calibration_20260528"

PAIRWISE_SOURCES = (
    (
        "combined_crossfield_support",
        BASE_RESULT_DIR
        / "leiden_multibasin_crossfield_budget12_support_20260519"
        / "combined_with_field30/signature_review/leiden_multibasin_pairwise_basin_matrix.csv",
    ),
    (
        "strict_field30_support",
        BASE_RESULT_DIR
        / "leiden_multibasin_signature_field30_budget12_support_20260519"
        / "signature_review/leiden_multibasin_pairwise_basin_matrix.csv",
    ),
    (
        "strict_field26_budget15_support",
        BASE_RESULT_DIR
        / "leiden_multibasin_signature_field26_citation_embedding_budget15_support_20260519"
        / "signature_review/leiden_multibasin_pairwise_basin_matrix.csv",
    ),
)

CANDIDATE_ROOTS = (
    BASE_RESULT_DIR / "leiden_multibasin_crossfield_budget12_support_20260519",
    BASE_RESULT_DIR / "leiden_multibasin_signature_field30_budget12_support_20260519",
    BASE_RESULT_DIR
    / "leiden_multibasin_signature_field26_citation_embedding_budget15_support_20260519",
)

ENDPOINT_TAU = 0.02
SAME_SUPPORT_MAX = 0.5
DISTINCT_SUPPORT_MIN = 0.75

ENDPOINT_IDENTITY_ROWS = "endpoint_identity_rows.csv"
CANDIDATE_PAIR_RELATIONS = "candidate_pair_relation_rows.csv"
IDENTITY_PAIR_RELATIONS = "identity_pair_relation_rows.csv"
CALIBRATED_CASE_SUMMARY = "calibrated_basin_case_summary.csv"
WALL_CANDIDATE_PAIRS = "wall_candidate_pair_rows.csv"
ROUTE_JOIN_CANDIDATE_PAIRS = "route_join_candidate_pair_rows.csv"
SUMMARY_JSON = "basin_definition_calibration_summary.json"
REPORT_MD = "basin_definition_calibration_report.md"
CONFIG_JSON = "calibration_config.json"

QUALITY_LIKE_TOKENS = (
    "quality",
    "delta_q",
    "relative_delta",
    "material",
    "cost",
    "elapsed",
    "regret",
    "selected_by",
    "operator_success",
)

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default

def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default

def _fmt_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.10g}"

def _case_tail(case: str) -> str:
    marker = "20260514_"
    return case.split(marker, 1)[1] if marker in case else case

def _case_id(case: str, candidate_budget: int) -> str:
    return f"{_case_tail(case)}_budget{candidate_budget}"

def _case_field_method(case: str) -> tuple[str, str]:
    tail = _case_tail(case)
    parts = tail.split("_")
    field = parts[0] if parts else ""
    method = "_".join(parts[1:]) if len(parts) > 1 else ""
    return field, method

def _load_candidate_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in CANDIDATE_ROOTS:
        for path in sorted(root.glob("*/candidate_level_rows.csv")):
            frame = _read_csv(path)
            if frame.empty or "case" not in frame:
                continue
            frame = frame.copy()
            frame["source_artifact"] = _rel(path)
            frame["candidate_budget"] = pd.to_numeric(
                frame["candidate_budget"],
                errors="coerce",
            ).fillna(0).astype(int)
            frame["candidate_index"] = pd.to_numeric(
                frame["candidate_index"],
                errors="coerce",
            ).fillna(-1).astype(int)
            frame["case_id"] = frame.apply(
                lambda row: _case_id(str(row["case"]), int(row["candidate_budget"])),
                axis=1,
            )
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

def _load_pairwise_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_label, path in PAIRWISE_SOURCES:
        frame = _read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_label"] = source_label
        frame["source_artifact"] = _rel(path)
        frame["candidate_budget"] = pd.to_numeric(
            frame["candidate_budget"],
            errors="coerce",
        ).fillna(0).astype(int)
        for column in ("left_candidate_index", "right_candidate_index"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(-1).astype(int)
        frame["case_id"] = frame.apply(
            lambda row: _case_id(str(row["case"]), int(row["candidate_budget"])),
            axis=1,
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

def _endpoint_status_rows(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    if candidate_rows.empty:
        return pd.DataFrame()
    required = [
        "case_id",
        "case",
        "candidate_budget",
        "candidate_index",
        "p5_basin_signature",
        "p5_basin_changed_support_node_count",
        "p5_basin_changed_support_node_hash",
        "p5_basin_sketch_node_hash",
        "p5_basin_sketch_sample_size",
        "source_artifact",
    ]
    available = [column for column in required if column in candidate_rows.columns]
    frame = candidate_rows[available].copy()
    frame["endpoint_signature"] = frame.get("p5_basin_signature", "").fillna("").astype(str)
    frame["support_node_count"] = pd.to_numeric(
        frame.get("p5_basin_changed_support_node_count"),
        errors="coerce",
    ).fillna(0).astype(int)
    frame["endpoint_filter_status"] = "accepted"
    frame.loc[frame["endpoint_signature"].eq(""), "endpoint_filter_status"] = "excluded_missing_signature"
    frame.loc[frame["support_node_count"].le(0), "endpoint_filter_status"] = "excluded_zero_support"

    rows: list[dict[str, Any]] = []
    for case_id, group in frame.groupby("case_id", sort=True):
        accepted = group[group["endpoint_filter_status"].eq("accepted")].copy()
        identity_map: dict[str, str] = {}
        representative_map: dict[str, int] = {}
        for rank, (signature, sig_group) in enumerate(
            accepted.sort_values("candidate_index").groupby("endpoint_signature", sort=False),
            start=1,
        ):
            identity_map[str(signature)] = f"{case_id}:E{rank:03d}"
            representative_map[str(signature)] = int(sig_group["candidate_index"].min())
        duplicate_counts = accepted["endpoint_signature"].value_counts().to_dict()
        for _, row in group.sort_values("candidate_index").iterrows():
            signature = str(row["endpoint_signature"])
            identity_id = identity_map.get(signature, "")
            duplicate_count = int(duplicate_counts.get(signature, 0))
            status = str(row["endpoint_filter_status"])
            if status == "accepted" and duplicate_count > 1:
                status = "accepted_duplicate_identity_member"
            field, method = _case_field_method(str(row["case"]))
            rows.append(
                {
                    "case_id": case_id,
                    "case": str(row["case"]),
                    "field": field,
                    "method": method,
                    "candidate_budget": int(row["candidate_budget"]),
                    "candidate_index": int(row["candidate_index"]),
                    "endpoint_identity_id": identity_id,
                    "endpoint_signature": signature,
                    "endpoint_filter_status": status,
                    "support_node_count": int(row["support_node_count"]),
                    "support_node_hash": str(row.get("p5_basin_changed_support_node_hash", "")),
                    "sketch_node_hash": str(row.get("p5_basin_sketch_node_hash", "")),
                    "sketch_sample_size": _safe_int(row.get("p5_basin_sketch_sample_size")),
                    "identity_member_count": duplicate_count if identity_id else 0,
                    "representative_candidate_index": representative_map.get(signature, ""),
                    "source_artifact": str(row.get("source_artifact", "")),
                }
            )
    return pd.DataFrame(rows)

def _endpoint_lookup(endpoint_rows: pd.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in endpoint_rows.iterrows():
        lookup[(str(row["case_id"]), int(row["candidate_index"]))] = row.to_dict()
    return lookup

def _candidate_relation(row: pd.Series, left: dict[str, Any] | None, right: dict[str, Any] | None) -> tuple[str, str]:
    if left is None or right is None:
        return "missing_endpoint_row", "missing_endpoint_metadata"
    left_status = str(left.get("endpoint_filter_status", ""))
    right_status = str(right.get("endpoint_filter_status", ""))
    if left_status.startswith("excluded") or right_status.startswith("excluded"):
        return "excluded_hygiene", "one_or_both_endpoints_filtered"

    left_identity = str(left.get("endpoint_identity_id", ""))
    right_identity = str(right.get("endpoint_identity_id", ""))
    endpoint_distance = _safe_float(row.get("sample_coassignment_distance"))
    support_distance = _safe_float(row.get("coarse_support_distance"))
    if left_identity and left_identity == right_identity:
        return "same_endpoint_identity", "same_signature_after_filtering"
    if (
        math.isfinite(endpoint_distance)
        and math.isfinite(support_distance)
        and endpoint_distance <= ENDPOINT_TAU
        and support_distance <= SAME_SUPPORT_MAX
    ):
        return "same_support_local", "endpoint_near_and_support_near"
    if math.isfinite(support_distance) and support_distance >= DISTINCT_SUPPORT_MIN and left_identity != right_identity:
        return "distinct_support_local", "support_far_and_endpoint_identity_distinct"
    return "ambiguous_support_local", "middle_support_zone_or_missing_metric"

def _candidate_pair_rows(pairwise: pd.DataFrame, endpoint_rows: pd.DataFrame) -> pd.DataFrame:
    lookup = _endpoint_lookup(endpoint_rows)
    rows: list[dict[str, Any]] = []
    for _, row in pairwise.iterrows():
        case_id = str(row["case_id"])
        left_index = int(row["left_candidate_index"])
        right_index = int(row["right_candidate_index"])
        left = lookup.get((case_id, left_index))
        right = lookup.get((case_id, right_index))
        relation, reason = _candidate_relation(row, left, right)
        field, method = _case_field_method(str(row["case"]))
        rows.append(
            {
                "source_label": str(row["source_label"]),
                "case_id": case_id,
                "case": str(row["case"]),
                "field": field,
                "method": method,
                "candidate_budget": int(row["candidate_budget"]),
                "left_candidate_index": left_index,
                "right_candidate_index": right_index,
                "left_endpoint_identity_id": "" if left is None else str(left.get("endpoint_identity_id", "")),
                "right_endpoint_identity_id": "" if right is None else str(right.get("endpoint_identity_id", "")),
                "left_endpoint_status": "" if left is None else str(left.get("endpoint_filter_status", "")),
                "right_endpoint_status": "" if right is None else str(right.get("endpoint_filter_status", "")),
                "endpoint_distance_metric": "sample_coassignment_distance",
                "endpoint_distance": _fmt_float(_safe_float(row.get("sample_coassignment_distance"))),
                "endpoint_distance_threshold": ENDPOINT_TAU,
                "support_distance_metric": "coarse_support_distance",
                "support_distance": _fmt_float(_safe_float(row.get("coarse_support_distance"))),
                "support_distance_source": str(row.get("coarse_support_distance_source", "")),
                "same_support_max": SAME_SUPPORT_MAX,
                "distinct_support_min": DISTINCT_SUPPORT_MIN,
                "support_relation": relation,
                "relation_reason": reason,
                "source_artifact": str(row.get("source_artifact", "")),
            }
        )
    return pd.DataFrame(rows)

def _relation_priority(relations: set[str]) -> str:
    if "excluded_hygiene" in relations:
        return "excluded_hygiene"
    if "missing_endpoint_row" in relations:
        return "missing_endpoint_row"
    if relations <= {"same_endpoint_identity"}:
        return "same_endpoint_identity"
    if relations <= {"same_support_local"}:
        return "same_support_local"
    if relations <= {"distinct_support_local"}:
        return "distinct_support_local"
    if "ambiguous_support_local" in relations:
        return "ambiguous_support_local"
    if "same_support_local" in relations and "distinct_support_local" in relations:
        return "ambiguous_mixed_relation"
    return "ambiguous_support_local"

def _identity_pair_rows(candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    if candidate_pairs.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    accepted = candidate_pairs[
        ~candidate_pairs["support_relation"].isin({"excluded_hygiene", "missing_endpoint_row"})
    ].copy()
    accepted = accepted[
        accepted["left_endpoint_identity_id"].astype(str).ne("")
        & accepted["right_endpoint_identity_id"].astype(str).ne("")
    ]
    if accepted.empty:
        return pd.DataFrame()
    accepted["identity_left"] = accepted[
        ["left_endpoint_identity_id", "right_endpoint_identity_id"]
    ].min(axis=1)
    accepted["identity_right"] = accepted[
        ["left_endpoint_identity_id", "right_endpoint_identity_id"]
    ].max(axis=1)
    for (source_label, case_id, left_id, right_id), group in accepted.groupby(
        ["source_label", "case_id", "identity_left", "identity_right"],
        sort=True,
    ):
        if left_id == right_id:
            relation = "same_endpoint_identity"
        else:
            relation = _relation_priority(set(group["support_relation"].astype(str)))
        endpoint = pd.to_numeric(group["endpoint_distance"], errors="coerce")
        support = pd.to_numeric(group["support_distance"], errors="coerce")
        first = group.iloc[0]
        rows.append(
            {
                "source_label": source_label,
                "case_id": case_id,
                "field": str(first["field"]),
                "method": str(first["method"]),
                "candidate_budget": int(first["candidate_budget"]),
                "left_endpoint_identity_id": left_id,
                "right_endpoint_identity_id": right_id,
                "candidate_pair_count": len(group),
                "endpoint_distance_min": _fmt_float(float(endpoint.min())) if endpoint.notna().any() else "",
                "endpoint_distance_max": _fmt_float(float(endpoint.max())) if endpoint.notna().any() else "",
                "support_distance_min": _fmt_float(float(support.min())) if support.notna().any() else "",
                "support_distance_max": _fmt_float(float(support.max())) if support.notna().any() else "",
                "calibrated_relation": relation,
                "relation_notes": "identity-level aggregate over filtered candidate pairs",
            }
        )
    return pd.DataFrame(rows)

def _component_count(endpoint_ids: list[str], identity_pairs: pd.DataFrame) -> tuple[int, int]:
    parent = {node: node for node in endpoint_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    same_edges = identity_pairs[
        identity_pairs["calibrated_relation"].isin({"same_endpoint_identity", "same_support_local"})
    ]
    for _, row in same_edges.iterrows():
        left = str(row["left_endpoint_identity_id"])
        right = str(row["right_endpoint_identity_id"])
        if left in parent and right in parent:
            union(left, right)
    roots = [find(node) for node in endpoint_ids]
    largest = max((roots.count(root) for root in set(roots)), default=0)
    return len(set(roots)), largest

def _case_summary_rows(
    endpoint_rows: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    identity_pairs: pd.DataFrame,
    phase1_dir: Path,
) -> pd.DataFrame:
    landscape = _read_csv(phase1_dir / "landscape_case_index.csv")
    route_lookup: dict[str, dict[str, str]] = {}
    if not landscape.empty:
        for _, row in landscape.iterrows():
            route_lookup[str(row.get("case_id", ""))] = {
                "has_route_trace_source": str(row.get("has_route_trace_source", "")),
                "route_trace_source_dirs": str(row.get("route_trace_source_dirs", "")),
            }
    rows: list[dict[str, Any]] = []
    case_ids = sorted(set(endpoint_rows["case_id"]).union(candidate_pairs["case_id"]).union(identity_pairs["case_id"]))
    for case_id in case_ids:
        endpoints = endpoint_rows[endpoint_rows["case_id"].eq(case_id)]
        pairs = candidate_pairs[candidate_pairs["case_id"].eq(case_id)]
        id_pairs = identity_pairs[identity_pairs["case_id"].eq(case_id)]
        accepted = endpoints[
            endpoints["endpoint_filter_status"].isin({"accepted", "accepted_duplicate_identity_member"})
        ]
        endpoint_ids = sorted(set(accepted["endpoint_identity_id"].dropna().astype(str)) - {""})
        component_count, largest_component = _component_count(endpoint_ids, id_pairs)
        relation_counts = id_pairs["calibrated_relation"].value_counts().to_dict()
        hygiene_excluded = int(pairs["support_relation"].eq("excluded_hygiene").sum()) if not pairs.empty else 0
        ambiguous = int(relation_counts.get("ambiguous_support_local", 0) + relation_counts.get("ambiguous_mixed_relation", 0))
        distinct = int(relation_counts.get("distinct_support_local", 0))
        same = int(relation_counts.get("same_support_local", 0) + relation_counts.get("same_endpoint_identity", 0))
        first = endpoints.iloc[0] if not endpoints.empty else {}
        if hygiene_excluded:
            readiness = "calibrated_after_filtering"
        elif ambiguous:
            readiness = "definition_calibrated_with_ambiguous_pairs"
        elif distinct:
            readiness = "non_ambiguous_distinct_pairs_available"
        elif same:
            readiness = "same_pairs_only"
        else:
            readiness = "endpoint_inventory_only"
        wall_status = (
            "pair_level_wall_candidates_available"
            if distinct > 0
            else "blocked_no_distinct_pairs"
        )
        if ambiguous > 0:
            wall_status = "pair_level_only_ambiguous_relations_present"
        if not distinct and ambiguous:
            wall_status = "blocked_ambiguous_without_distinct_pairs"
        route_info = route_lookup.get(case_id, {})
        has_route_trace_source = route_info.get("has_route_trace_source", "")
        route_trace_source_dirs = route_info.get("route_trace_source_dirs", "")
        if distinct > 0 and has_route_trace_source == "yes":
            phase2_join_status = "route_join_candidate_pairs_available"
        elif has_route_trace_source == "yes":
            phase2_join_status = "route_sources_available_but_no_distinct_pairs"
        else:
            phase2_join_status = "no_route_trace_source"
        rows.append(
            {
                "case_id": case_id,
                "field": str(first.get("field", "")) if isinstance(first, pd.Series) else "",
                "method": str(first.get("method", "")) if isinstance(first, pd.Series) else "",
                "candidate_budget": _safe_int(first.get("candidate_budget", "")) if isinstance(first, pd.Series) else "",
                "raw_endpoint_rows": len(endpoints),
                "accepted_endpoint_rows": len(accepted),
                "accepted_endpoint_identity_count": len(endpoint_ids),
                "filtered_endpoint_rows": int(len(endpoints) - len(accepted)),
                "calibrated_support_local_component_count": component_count,
                "largest_support_local_component_size": largest_component,
                "identity_pair_count": len(id_pairs),
                "same_identity_or_support_pair_count": same,
                "ambiguous_identity_pair_count": ambiguous,
                "distinct_identity_pair_count": distinct,
                "hygiene_excluded_candidate_pair_count": hygiene_excluded,
                "has_route_trace_source": has_route_trace_source,
                "route_trace_source_dirs": route_trace_source_dirs,
                "basin_definition_status": readiness,
                "wall_readiness_status": wall_status,
                "phase2_route_join_status": phase2_join_status,
            }
        )
    return pd.DataFrame(rows)

def _wall_candidate_pairs(identity_pairs: pd.DataFrame, case_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if identity_pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    out = identity_pairs[
        identity_pairs["calibrated_relation"].eq("distinct_support_local")
    ].copy()
    if out.empty:
        return out, out.copy()
    route_cols = case_summary[
        [
            "case_id",
            "has_route_trace_source",
            "route_trace_source_dirs",
            "phase2_route_join_status",
        ]
    ].copy()
    out = out.merge(route_cols, on="case_id", how="left")
    out["wall_assignment_status"] = "candidate_pair_only_no_wall_claim"
    out["wall_evidence_allowed"] = "yes_after_route_join"
    route_join = out[out["phase2_route_join_status"].eq("route_join_candidate_pairs_available")].copy()
    return out, route_join

def _quality_column_leaks(frames: dict[str, pd.DataFrame]) -> list[str]:
    leaks: list[str] = []
    for name, frame in frames.items():
        for column in frame.columns:
            lower = column.lower()
            if any(token in lower for token in QUALITY_LIKE_TOKENS):
                leaks.append(f"{name}:{column}")
    return leaks

def _write_report(path: Path, summary: dict[str, Any], case_summary: pd.DataFrame) -> None:
    lines = [
        "# Leiden Basin Definition Calibration",
        "",
        "Status: calibration pass before wall cartography",
        "Date: 2026-05-28",
        "",
        "This report calibrates support-local basin relations without using quality, materiality, cost, ranking, or operator-success fields.",
        "",
        "## Decision",
        "",
        "Wall cartography remains blocked as a case-level claim. The useful next object is a pair-level inventory of non-ambiguous support-local relations.",
        "",
        "Rules used here:",
        "",
        f"- endpoint identity: filtered `p5_basin_signature` within a fixed case.",
        f"- same support-local relation: endpoint distance <= {ENDPOINT_TAU} and support distance <= {SAME_SUPPORT_MAX}.",
        f"- distinct support-local relation: endpoint identities differ and support distance >= {DISTINCT_SUPPORT_MIN}.",
        f"- ambiguous support-local relation: middle support zone, missing metric, or mixed relation after identity aggregation.",
        "- field34 zero-support endpoints are filtered before identity aggregation.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in (
        "case_count",
        "raw_endpoint_rows",
        "accepted_endpoint_rows",
        "accepted_endpoint_identity_count",
        "candidate_pair_rows",
        "identity_pair_rows",
        "same_identity_pair_rows",
        "same_support_pair_rows",
        "ambiguous_identity_pair_rows",
        "distinct_identity_pair_rows",
        "wall_candidate_pair_rows",
        "route_join_candidate_pair_rows",
        "cases_with_distinct_pairs",
        "cases_with_ambiguous_pairs",
        "cases_with_filtered_endpoints",
        "cases_with_route_join_candidates",
    ):
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(
        [
            "",
        "## Case Readiness",
        "",
        "| case_id | identities | components | same | ambiguous | distinct | route_source | status | wall_status | phase2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for _, row in case_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_id"]),
                    str(row["accepted_endpoint_identity_count"]),
                    str(row["calibrated_support_local_component_count"]),
                    str(row["same_identity_or_support_pair_count"]),
                    str(row["ambiguous_identity_pair_count"]),
                    str(row["distinct_identity_pair_count"]),
                    str(row["has_route_trace_source"]),
                    str(row["basin_definition_status"]),
                    str(row["wall_readiness_status"]),
                    str(row["phase2_route_join_status"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The calibration produces pair-level distinct support-local candidates, not wall claims.",
            "- Ambiguous middle-zone pairs are common enough that basin definition should remain a first-class research object.",
            "- Phase 2 should join route or wall evidence only to `distinct_support_local` identity pairs and keep ambiguous pairs out of wall claims.",
            "- The route-join candidate table is the narrow next input for wall evidence review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(phase1_dir: Path, review_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidate_rows()
    pairwise = _load_pairwise_rows()
    endpoint_rows = _endpoint_status_rows(candidates)
    candidate_pairs = _candidate_pair_rows(pairwise, endpoint_rows)
    identity_pairs = _identity_pair_rows(candidate_pairs)
    case_summary = _case_summary_rows(endpoint_rows, candidate_pairs, identity_pairs, phase1_dir)
    wall_pairs, route_join_pairs = _wall_candidate_pairs(identity_pairs, case_summary)

    frames = {
        ENDPOINT_IDENTITY_ROWS: endpoint_rows,
        CANDIDATE_PAIR_RELATIONS: candidate_pairs,
        IDENTITY_PAIR_RELATIONS: identity_pairs,
        CALIBRATED_CASE_SUMMARY: case_summary,
        WALL_CANDIDATE_PAIRS: wall_pairs,
        ROUTE_JOIN_CANDIDATE_PAIRS: route_join_pairs,
    }
    leaks = _quality_column_leaks(frames)
    if leaks:
        raise ValueError("quality-like columns leaked into calibration outputs: " + ", ".join(leaks))

    summary = {
        "status": "basin_definition_calibration",
        "date": "2026-05-28",
        "phase1_dir": _rel(phase1_dir),
        "review_dir": _rel(review_dir),
        "endpoint_tau": ENDPOINT_TAU,
        "same_support_max": SAME_SUPPORT_MAX,
        "distinct_support_min": DISTINCT_SUPPORT_MIN,
        "case_count": int(case_summary["case_id"].nunique()) if not case_summary.empty else 0,
        "raw_endpoint_rows": int(len(endpoint_rows)),
        "accepted_endpoint_rows": int(
            endpoint_rows["endpoint_filter_status"].isin({"accepted", "accepted_duplicate_identity_member"}).sum()
        )
        if not endpoint_rows.empty
        else 0,
        "accepted_endpoint_identity_count": int(
            endpoint_rows["endpoint_identity_id"].replace("", pd.NA).dropna().nunique()
        )
        if not endpoint_rows.empty
        else 0,
        "candidate_pair_rows": int(len(candidate_pairs)),
        "identity_pair_rows": int(len(identity_pairs)),
        "same_identity_pair_rows": int(identity_pairs["calibrated_relation"].eq("same_endpoint_identity").sum())
        if not identity_pairs.empty
        else 0,
        "same_support_pair_rows": int(identity_pairs["calibrated_relation"].eq("same_support_local").sum())
        if not identity_pairs.empty
        else 0,
        "ambiguous_identity_pair_rows": int(
            identity_pairs["calibrated_relation"].isin({"ambiguous_support_local", "ambiguous_mixed_relation"}).sum()
        )
        if not identity_pairs.empty
        else 0,
        "distinct_identity_pair_rows": int(identity_pairs["calibrated_relation"].eq("distinct_support_local").sum())
        if not identity_pairs.empty
        else 0,
        "wall_candidate_pair_rows": int(len(wall_pairs)),
        "route_join_candidate_pair_rows": int(len(route_join_pairs)),
        "cases_with_distinct_pairs": int(case_summary["distinct_identity_pair_count"].gt(0).sum())
        if not case_summary.empty
        else 0,
        "cases_with_ambiguous_pairs": int(case_summary["ambiguous_identity_pair_count"].gt(0).sum())
        if not case_summary.empty
        else 0,
        "cases_with_filtered_endpoints": int(case_summary["filtered_endpoint_rows"].gt(0).sum())
        if not case_summary.empty
        else 0,
        "cases_with_route_join_candidates": int(
            case_summary["phase2_route_join_status"].eq("route_join_candidate_pairs_available").sum()
        )
        if not case_summary.empty
        else 0,
        "claim_boundary": (
            "Calibration defines endpoint and support-local relations only; no wall, quality, "
            "cost, materiality, ranking, or operator-success claim is made."
        ),
    }

    for filename, frame in frames.items():
        _write_csv(frame, output_dir / filename)
    config = {
        "script": _rel(Path(__file__)),
        "phase1_dir": _rel(phase1_dir),
        "review_dir": _rel(review_dir),
        "pairwise_sources": [{"label": label, "path": _rel(path)} for label, path in PAIRWISE_SOURCES],
        "candidate_roots": [_rel(path) for path in CANDIDATE_ROOTS],
        "endpoint_tau": ENDPOINT_TAU,
        "same_support_max": SAME_SUPPORT_MAX,
        "distinct_support_min": DISTINCT_SUPPORT_MIN,
        "quality_fields_excluded": True,
    }
    (output_dir / CONFIG_JSON).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / REPORT_MD, summary, case_summary)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run(args.phase1_dir, args.review_dir, args.output_dir)
    print(json.dumps({"output_dir": _rel(args.output_dir), **summary}, indent=2))

if __name__ == "__main__":
    main()
