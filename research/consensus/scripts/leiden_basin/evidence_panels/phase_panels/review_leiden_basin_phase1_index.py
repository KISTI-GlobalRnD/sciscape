#!/usr/bin/env python3
"""Review Phase 1 Leiden basin indexes before wall analysis.

This script runs basin-only consistency and sensitivity checks. It does not use
quality, materiality, cost, ranking, or operator-success values to decide basin
identity.
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
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_phase1_review_20260528"

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

QUALITY_LIKE_TOKENS = (
    "quality",
    "delta_q",
    "relative_delta",
    "material",
    "cost",
    "regret",
    "selected_by",
    "operator_success",
)

SUPPORT_TAUS = (0.25, 0.5, 0.75, 1.0)
ENDPOINT_TAU = 0.02
SAME_SUPPORT_MAX = 0.5
DISTINCT_SUPPORT_MIN = 0.75

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

def _case_tail(case: str) -> str:
    marker = "20260514_"
    return case.split(marker, 1)[1] if marker in case else case

def _case_id(case: str, candidate_budget: int) -> str:
    return f"{_case_tail(case)}_budget{candidate_budget}"

def _component_count(pair_rows: pd.DataFrame, *, support_tau: float) -> int:
    if pair_rows.empty:
        return 0
    nodes = set(pair_rows["left_candidate_index"]).union(pair_rows["right_candidate_index"])
    parent = {node: node for node in nodes}

    def find(item: Any) -> Any:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: Any, right: Any) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for _, row in pair_rows.iterrows():
        endpoint_distance = float(row.get("sample_coassignment_distance", math.inf))
        support_distance = float(row.get("coarse_support_distance", math.inf))
        if endpoint_distance <= ENDPOINT_TAU and support_distance <= support_tau:
            union(row["left_candidate_index"], row["right_candidate_index"])
    return len({find(node) for node in nodes})

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
        frame["case_id"] = frame.apply(
            lambda row: _case_id(str(row["case"]), int(row["candidate_budget"])),
            axis=1,
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

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
            frame["case_id"] = frame.apply(
                lambda row: _case_id(str(row["case"]), int(row["candidate_budget"])),
                axis=1,
            )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

def _quality_column_leaks(phase1_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(phase1_dir.glob("*.csv")):
        frame = _read_csv(path)
        for column in frame.columns:
            lower = column.lower()
            if any(token in lower for token in QUALITY_LIKE_TOKENS):
                rows.append(
                    {
                        "file": _rel(path),
                        "column": column,
                        "status": "leak",
                    }
                )
    return pd.DataFrame(rows, columns=["file", "column", "status"])

def _consistency_checks(phase1_dir: Path, pairwise: pd.DataFrame) -> pd.DataFrame:
    landscape = _read_csv(phase1_dir / "landscape_case_index.csv")
    hygiene = _read_csv(phase1_dir / "metric_hygiene_audit.csv")
    cartography = _read_csv(phase1_dir / "basin_cartography_case_index.csv")
    wall = _read_csv(phase1_dir / "wall_evidence_rows.csv")
    route = _read_csv(phase1_dir / "route_taxonomy_rows.csv")
    summary_path = phase1_dir / "basin_cartography_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    leaks = _quality_column_leaks(phase1_dir)

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any, note: str = "") -> None:
        checks.append(
            {
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "note": note,
            }
        )

    add("landscape_case_count_matches_summary", len(landscape) == summary.get("case_count"), len(landscape), summary.get("case_count"))
    add("hygiene_case_count_matches_landscape", len(hygiene) == len(landscape), len(hygiene), len(landscape))
    add("cartography_case_count_matches_landscape", len(cartography) == len(landscape), len(cartography), len(landscape))
    add("wall_route_inventory_counts_match", len(wall) == len(route), len(wall), len(route))
    add("no_quality_like_columns_in_phase1_outputs", leaks.empty, len(leaks), 0)
    add(
        "all_global_basin_assignments_unresolved",
        set(cartography.get("global_assignment_status", [])) == {"unresolved"},
        sorted(set(cartography.get("global_assignment_status", []))),
        ["unresolved"],
    )
    add(
        "support_count_never_exceeds_endpoint_identity_count",
        bool((cartography["support_local_basin_count"] <= cartography["endpoint_identity_count"]).all()),
        "max_difference="
        + str((cartography["support_local_basin_count"] - cartography["endpoint_identity_count"]).max()),
        "<=0",
    )
    needs_filtering = hygiene["metric_hygiene_status"].eq("needs_filtering")
    hygiene_problem = hygiene["zero_support_rows"].gt(0) | hygiene["duplicate_endpoint_rows"].gt(0)
    add(
        "needs_filtering_matches_zero_or_duplicate_rows",
        bool((needs_filtering == hygiene_problem).all()),
        int(needs_filtering.sum()),
        int(hygiene_problem.sum()),
    )

    mismatches = []
    for case_id, rows in pairwise.groupby("case_id"):
        indexed = cartography.loc[
            cartography["case_id"].eq(case_id),
            "support_local_basin_count",
        ]
        if indexed.empty:
            mismatches.append(f"{case_id}:missing")
            continue
        computed = _component_count(rows, support_tau=0.5)
        if computed != int(indexed.iloc[0]):
            mismatches.append(f"{case_id}:{computed}!={int(indexed.iloc[0])}")
    add("support_tau_0p5_recomputes_indexed_group_counts", not mismatches, ";".join(mismatches), "no mismatches")

    return pd.DataFrame(checks)

def _threshold_sensitivity(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if pairwise.empty:
        return pd.DataFrame()
    for (source_label, case_id), group in pairwise.groupby(["source_label", "case_id"]):
        endpoints = len(set(group["left_candidate_index"]).union(group["right_candidate_index"]))
        row: dict[str, Any] = {
            "source_label": source_label,
            "case_id": case_id,
            "endpoint_candidate_count": endpoints,
        }
        for support_tau in SUPPORT_TAUS:
            row[f"support_groups_tau_{str(support_tau).replace('.', 'p')}"] = _component_count(
                group,
                support_tau=support_tau,
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_label", "case_id"])

def _trizone_counts(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if pairwise.empty:
        return pd.DataFrame()
    for (source_label, case_id), group in pairwise.groupby(["source_label", "case_id"]):
        support = pd.to_numeric(group["coarse_support_distance"], errors="coerce")
        same = int(support.le(SAME_SUPPORT_MAX).sum())
        ambiguous = int((support.gt(SAME_SUPPORT_MAX) & support.lt(DISTINCT_SUPPORT_MIN)).sum())
        distinct = int(support.ge(DISTINCT_SUPPORT_MIN).sum())
        total = len(group)
        rows.append(
            {
                "source_label": source_label,
                "case_id": case_id,
                "pair_count": total,
                "same_pair_count_support_le_0p5": same,
                "ambiguous_pair_count_support_0p5_to_0p75": ambiguous,
                "distinct_pair_count_support_ge_0p75": distinct,
                "same_pair_fraction": same / total if total else math.nan,
                "ambiguous_pair_fraction": ambiguous / total if total else math.nan,
                "distinct_pair_fraction": distinct / total if total else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["source_label", "case_id"])

def _field34_filtering(candidate_rows: pd.DataFrame, phase1_dir: Path) -> pd.DataFrame:
    cartography = _read_csv(phase1_dir / "basin_cartography_case_index.csv")
    rows: list[dict[str, Any]] = []
    for case_id, group in candidate_rows.groupby("case_id"):
        if not case_id.startswith("field34_"):
            continue
        support_counts = pd.to_numeric(
            group.get("p5_basin_changed_support_node_count"),
            errors="coerce",
        ).fillna(0)
        raw_rows = len(group)
        raw_identities = group["p5_basin_signature"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()
        zero_support = int(support_counts.eq(0).sum())
        filtered = group[support_counts.gt(0)].copy()
        filtered_rows = len(filtered)
        filtered_identities = (
            filtered["p5_basin_signature"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()
            if not filtered.empty
            else 0
        )
        indexed_support = cartography.loc[
            cartography["case_id"].eq(case_id),
            "support_local_basin_count",
        ]
        rows.append(
            {
                "case_id": case_id,
                "raw_endpoint_rows": raw_rows,
                "raw_endpoint_identities": raw_identities,
                "indexed_support_local_groups": int(indexed_support.iloc[0])
                if not indexed_support.empty
                else "",
                "zero_support_rows": zero_support,
                "filtered_endpoint_rows": filtered_rows,
                "filtered_endpoint_identities": filtered_identities,
                "filtering_effect": raw_identities - filtered_identities,
                "recommended_phase1_status": "filter_before_basin_count"
                if zero_support or raw_identities != filtered_identities
                else "usable",
            }
        )
    return pd.DataFrame(rows).sort_values("case_id")

def _write_report(
    path: Path,
    *,
    checks: pd.DataFrame,
    threshold: pd.DataFrame,
    trizone: pd.DataFrame,
    field34: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Leiden Basin Phase 1 Review",
        "",
        "Status: basin-only review after Phase 1 index generation",
        "Date: 2026-05-28",
        "",
        "This review tests the Phase 1 basin index without using quality, materiality, cost, ranking, or operator-success fields.",
        "",
        "## Consistency Checks",
        "",
        "| status | check | observed | expected |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in checks.iterrows():
        lines.append(
            f"| {row['status']} | {row['check_name']} | {row['observed']} | {row['expected']} |"
        )

    lines.extend(
        [
            "",
            "## Threshold Stress Result",
            "",
            "The current `support_tau=0.5` index is internally reproducible, but it is not stable enough to be treated as a final basin definition.",
            "",
            f"- total pair rows reviewed: {summary['total_pair_rows']}",
            f"- pairs classified same at support <= 0.5: {summary['same_pairs']}",
            f"- pairs classified ambiguous at 0.5 < support < 0.75: {summary['ambiguous_pairs']}",
            f"- pairs classified distinct at support >= 0.75: {summary['distinct_pairs']}",
            "",
            "This argues for a three-zone rule before wall analysis.",
            "",
            "## Field34 Hygiene",
            "",
            "| case_id | raw rows | raw identities | zero-support rows | filtered rows | filtered identities | status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in field34.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_id"]),
                    str(row["raw_endpoint_rows"]),
                    str(row["raw_endpoint_identities"]),
                    str(row["zero_support_rows"]),
                    str(row["filtered_endpoint_rows"]),
                    str(row["filtered_endpoint_identities"]),
                    str(row["recommended_phase1_status"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Direction Decision",
            "",
            "Decision: do not proceed directly to wall cartography yet.",
            "",
            "The next step should be a basin-definition calibration pass:",
            "",
            "1. Keep `endpoint_identity` accepted for clean field12, field26, and field30 rows.",
            "2. Treat `support_tau=0.5` as a strict inventory threshold, not a final basin definition.",
            "3. Introduce a three-zone support-local relation rule: same, distinct, ambiguous.",
            "4. Filter field34 zero-support and duplicate endpoints before using field34 as a basin case.",
            "5. Keep global observed basin unresolved until the global endpoint-distance rule is accepted.",
            "6. Promote wall evidence only for basin pairs whose source/target assignment is not ambiguous.",
            "",
            "Recommended provisional rule to test next:",
            "",
            "- `same_support_local`: endpoint distance <= 0.02 and support distance <= 0.5.",
            "- `distinct_support_local`: support distance >= 0.75, with endpoint identity already distinct.",
            "- `ambiguous_support_local`: 0.5 < support distance < 0.75, or any field34 no-op/duplicate hygiene issue.",
            "",
            "This keeps the research basin-first and prevents threshold choice from silently becoming a basin claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(phase1_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise = _load_pairwise_rows()
    candidates = _load_candidate_rows()
    checks = _consistency_checks(phase1_dir, pairwise)
    threshold = _threshold_sensitivity(pairwise)
    trizone = _trizone_counts(pairwise)
    field34 = _field34_filtering(candidates, phase1_dir)

    summary = {
        "phase1_dir": _rel(phase1_dir),
        "total_pair_rows": int(trizone["pair_count"].sum()) if not trizone.empty else 0,
        "same_pairs": int(trizone["same_pair_count_support_le_0p5"].sum()) if not trizone.empty else 0,
        "ambiguous_pairs": int(trizone["ambiguous_pair_count_support_0p5_to_0p75"].sum()) if not trizone.empty else 0,
        "distinct_pairs": int(trizone["distinct_pair_count_support_ge_0p75"].sum()) if not trizone.empty else 0,
        "failed_consistency_checks": int(checks["status"].eq("fail").sum()) if not checks.empty else 0,
        "field34_cases_needing_filtering": int(
            field34["recommended_phase1_status"].eq("filter_before_basin_count").sum()
        )
        if not field34.empty
        else 0,
        "decision": "definition_calibration_before_wall_cartography",
    }

    _write_csv(checks, output_dir / "phase1_consistency_checks.csv")
    _write_csv(threshold, output_dir / "support_threshold_sensitivity.csv")
    _write_csv(trizone, output_dir / "support_pair_trizone_counts.csv")
    _write_csv(field34, output_dir / "field34_filtering_review.csv")
    (output_dir / "phase1_review_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / "phase1_review_report.md",
        checks=checks,
        threshold=threshold,
        trizone=trizone,
        field34=field34,
        summary=summary,
    )
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run(args.phase1_dir, args.output_dir)
    print(json.dumps({"output_dir": _rel(args.output_dir), **summary}, indent=2))

if __name__ == "__main__":
    main()
