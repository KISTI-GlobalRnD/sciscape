#!/usr/bin/env python3
"""Refine stable ambiguous Leiden basin relations with cached memberships.

This script is diagnostic-only. It inspects stable route-evidence rows whose
basin relation is still ambiguous, and asks whether cached full memberships add
stronger identity evidence. It does not run routes, tune operators, or evaluate
basin value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sciscape.clustering.leiden_basin_profile import (
    changed_support_nodes,
    endpoint_distance,
    support_distance,
)


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_COVERAGE_DIR = BASE_RESULT_DIR / "leiden_basin_wall_panel_context_coverage_20260528"
DEFAULT_ENDPOINT_CACHE_DIR = BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_endpoint_cache_20260528"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_stable_ambiguous_relation_refinement_20260528"

AMBIGUOUS_QUEUE_CSV = "ambiguous_relation_refinement_queue.csv"
COVERAGE_ROWS_CSV = "wall_panel_context_coverage_rows.csv"

REFINEMENT_ROWS_CSV = "stable_ambiguous_relation_refinement_rows.csv"
ENDPOINT_CACHE_LINKS_CSV = "stable_ambiguous_endpoint_cache_links.csv"
SUMMARY_JSON = "stable_ambiguous_relation_refinement_summary.json"
REPORT_MD = "stable_ambiguous_relation_refinement_report.md"
CONFIG_JSON = "stable_ambiguous_relation_refinement_config.json"

SAME_SUPPORT_MAX = 0.5
DISTINCT_SUPPORT_MIN = 0.75
NEAR_SAME_MARGIN = 0.02
NEAR_DISTINCT_MARGIN = 0.005
COASSIGNMENT_SAMPLE_MAX = 4096


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
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _stable_sample_nodes(nodes: np.ndarray, max_nodes: int) -> np.ndarray:
    unique = np.unique(np.asarray(nodes, dtype=np.uint32))
    if max_nodes <= 0 or unique.size <= max_nodes:
        return unique
    scored: list[tuple[bytes, int]] = []
    for node in unique:
        payload = np.asarray([int(node)], dtype=np.uint64).tobytes()
        scored.append((hashlib.blake2b(payload, digest_size=8).digest(), int(node)))
    selected = sorted(scored)[:max_nodes]
    return np.asarray(sorted(node for _digest, node in selected), dtype=np.uint32)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _membership_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.membership.npy"


def _metadata_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.metadata.json"


def _load_cache_index(cache_dir: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    endpoint_index: dict[tuple[str, int], dict[str, Any]] = {}
    baseline_index: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(cache_dir.glob("*.metadata.json")):
        metadata = _load_json(metadata_path)
        cache_key = str(metadata.get("cache_key", metadata_path.stem))
        metadata["_metadata_path"] = _rel(metadata_path)
        metadata["_membership_path"] = _rel(_membership_path(cache_dir, cache_key))
        kind = str(metadata.get("kind", ""))
        case_id = str(metadata.get("case_id", ""))
        if kind == "baseline":
            baseline_index[cache_key] = metadata
        elif kind == "endpoint":
            candidate_index = _safe_int(metadata.get("candidate_index"))
            if candidate_index is not None:
                endpoint_index[(case_id, candidate_index)] = metadata
    return endpoint_index, baseline_index


def _load_membership(cache_dir: Path, metadata: dict[str, Any]) -> np.ndarray | None:
    cache_key = str(metadata.get("cache_key", ""))
    path = _membership_path(cache_dir, cache_key)
    if not path.exists():
        return None
    return np.asarray(np.load(path), dtype=np.uint64)


def _refinement_status(
    *,
    left_hash: str,
    right_hash: str,
    exact_support_distance: float,
) -> tuple[str, str]:
    if left_hash and right_hash and left_hash == right_hash:
        return "confirmed_same_observed_basin", "endpoint membership hashes match"
    if math.isfinite(exact_support_distance) and exact_support_distance <= SAME_SUPPORT_MAX:
        return "same_support_local_under_current_rule", "exact cached support distance is in same zone"
    if math.isfinite(exact_support_distance) and exact_support_distance >= DISTINCT_SUPPORT_MIN:
        return "distinct_support_local_under_current_rule", "exact cached support distance is in distinct zone"
    if math.isfinite(exact_support_distance) and DISTINCT_SUPPORT_MIN - exact_support_distance <= NEAR_DISTINCT_MARGIN:
        return (
            "near_distinct_boundary_requires_definition_choice",
            "exact cached support distance is just below the distinct threshold",
        )
    if math.isfinite(exact_support_distance) and exact_support_distance - SAME_SUPPORT_MAX <= NEAR_SAME_MARGIN:
        return (
            "near_same_boundary_requires_definition_choice",
            "exact cached support distance is just above the same threshold",
        )
    return "ambiguous_middle_zone_unresolved", "exact cached support evidence remains between thresholds"


def _route_promotion_status(refinement_status: str) -> str:
    if refinement_status == "distinct_support_local_under_current_rule":
        return "eligible_for_route_gate_recheck_not_promoted_here"
    return "blocked_until_basin_relation_rule_fixed"


def _analyze_pair(
    row: pd.Series,
    cache_dir: Path,
    endpoint_index: dict[tuple[str, int], dict[str, Any]],
    baseline_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = str(row["case_id"])
    left_index = _safe_int(row.get("left_representative_candidate_index"))
    right_index = _safe_int(row.get("right_representative_candidate_index"))
    if left_index is None or right_index is None:
        raise ValueError(f"missing candidate indices for {row['panel_pair_id']}")
    left_meta = endpoint_index.get((case_id, left_index))
    right_meta = endpoint_index.get((case_id, right_index))
    if left_meta is None or right_meta is None:
        raise ValueError(f"missing endpoint cache metadata for {row['panel_pair_id']}")
    left_baseline_key = str(left_meta.get("baseline_cache_key", ""))
    right_baseline_key = str(right_meta.get("baseline_cache_key", ""))
    baseline_meta = baseline_index.get(left_baseline_key)
    if baseline_meta is None:
        raise ValueError(f"missing baseline cache metadata for {row['panel_pair_id']}")
    if right_baseline_key and right_baseline_key != left_baseline_key:
        raise ValueError(f"endpoint baselines differ for {row['panel_pair_id']}")

    baseline = _load_membership(cache_dir, baseline_meta)
    left_membership = _load_membership(cache_dir, left_meta)
    right_membership = _load_membership(cache_dir, right_meta)
    if baseline is None or left_membership is None or right_membership is None:
        raise ValueError(f"missing membership arrays for {row['panel_pair_id']}")

    left_support = changed_support_nodes(baseline, left_membership)
    right_support = changed_support_nodes(baseline, right_membership)
    exact_support_distance, support_intersection, support_union = support_distance(
        left_support,
        right_support,
    )
    left_right_changed = changed_support_nodes(left_membership, right_membership)
    support_union_nodes = np.union1d(left_support, right_support).astype(np.uint32)
    coassignment_nodes = _stable_sample_nodes(support_union_nodes, COASSIGNMENT_SAMPLE_MAX)
    support_union_endpoint_distance = endpoint_distance(
        left_membership,
        right_membership,
        coassignment_nodes,
    )
    left_hash = str(left_meta.get("membership_hash", ""))
    right_hash = str(right_meta.get("membership_hash", ""))
    refinement_status, status_reason = _refinement_status(
        left_hash=left_hash,
        right_hash=right_hash,
        exact_support_distance=exact_support_distance,
    )
    calibration_support = _safe_float(row.get("support_distance_max"))
    same_margin = exact_support_distance - SAME_SUPPORT_MAX
    distinct_margin = DISTINCT_SUPPORT_MIN - exact_support_distance
    result = {
        "panel_pair_id": str(row["panel_pair_id"]),
        "field": str(row["field"]),
        "case_id": case_id,
        "method": str(row["method"]),
        "ambiguous_band": str(row.get("ambiguous_band", "")),
        "left_candidate_index": left_index,
        "right_candidate_index": right_index,
        "left_endpoint_identity_id": str(row["left_endpoint_identity_id"]),
        "right_endpoint_identity_id": str(row["right_endpoint_identity_id"]),
        "left_membership_hash": left_hash,
        "right_membership_hash": right_hash,
        "membership_hash_match": bool(left_hash and left_hash == right_hash),
        "baseline_cache_key": left_baseline_key,
        "left_support_node_count_exact": int(left_support.size),
        "right_support_node_count_exact": int(right_support.size),
        "exact_support_intersection": int(support_intersection),
        "exact_support_union": int(support_union),
        "exact_support_distance": exact_support_distance,
        "calibration_support_distance": calibration_support,
        "support_distance_delta_from_calibration": (
            exact_support_distance - calibration_support
            if math.isfinite(calibration_support)
            else math.nan
        ),
        "same_threshold_margin": same_margin,
        "distinct_threshold_margin": distinct_margin,
        "left_right_label_aligned_changed_node_count": int(left_right_changed.size),
        "left_right_label_aligned_changed_fraction": float(left_right_changed.size)
        / float(left_membership.size),
        "support_union_coassignment_sample_size": int(coassignment_nodes.size),
        "support_union_endpoint_distance": support_union_endpoint_distance,
        "stable_route_evidence_status": str(row.get("existing_wall_claim_gate_status", "")),
        "relation_refinement_need": str(row.get("relation_refinement_need", "")),
        "identity_refinement_status": refinement_status,
        "identity_refinement_reason": status_reason,
        "route_promotion_status": _route_promotion_status(refinement_status),
        "evidence_grade": "cached_full_membership_exact_support",
        "claim_boundary": (
            "Basin relation refinement only; no route execution, wall promotion, "
            "or basin evaluation is made."
        ),
    }
    cache_links = []
    for role, metadata in (
        ("baseline", baseline_meta),
        ("left_endpoint", left_meta),
        ("right_endpoint", right_meta),
    ):
        cache_links.append(
            {
                "panel_pair_id": str(row["panel_pair_id"]),
                "cache_role": role,
                "case_id": case_id,
                "candidate_index": "" if role == "baseline" else metadata.get("candidate_index", ""),
                "cache_key": metadata.get("cache_key", ""),
                "membership_path": metadata.get("_membership_path", ""),
                "metadata_path": metadata.get("_metadata_path", ""),
            }
        )
    return result, cache_links


def _write_report(path: Path, summary: dict[str, Any], rows: pd.DataFrame) -> None:
    lines = [
        "# Stable Ambiguous Basin Relation Refinement",
        "",
        "Status: stable ambiguous relation refinement prepared",
        "Date: 2026-05-28",
        "",
        "This artifact inspects cached full memberships for stable route-evidence rows whose basin relation remains ambiguous. It does not run routes, promote wall claims, or evaluate basin value.",
        "",
        "## Summary",
        "",
        f"- stable ambiguous input pairs: {summary['stable_ambiguous_input_pair_count']}",
        f"- refined pair rows: {summary['refined_pair_count']}",
        f"- route-promotion eligible rows: {summary['route_promotion_eligible_count']}",
        "",
        "## Refinement Status",
        "",
        "| status | pairs |",
        "| --- | ---: |",
    ]
    for status, count in sorted(summary["identity_refinement_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Pair Details",
            "",
            "| pair_id | band | exact_support_distance | same_margin | distinct_margin | refinement_status | route_status |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in rows.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["ambiguous_band"]),
                    f"{float(row['exact_support_distance']):.6f}",
                    f"{float(row['same_threshold_margin']):.6f}",
                    f"{float(row['distinct_threshold_margin']):.6f}",
                    str(row["identity_refinement_status"]),
                    str(row["route_promotion_status"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- None of the stable ambiguous rows is promoted to supported wall evidence here.",
            "- Near-distinct rows are definition-boundary cases, not route-runner failures.",
            "- The near-same row should be handled by the same-zone boundary rule before any route expansion.",
            "- The next method decision is whether support-local basin relation should stay at hard thresholds or gain a boundary-review class backed by cached membership evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(coverage_dir: Path, endpoint_cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ambiguous = _read_csv(coverage_dir / AMBIGUOUS_QUEUE_CSV)
    coverage = _read_csv(coverage_dir / COVERAGE_ROWS_CSV)
    if ambiguous.empty:
        raise FileNotFoundError(coverage_dir / AMBIGUOUS_QUEUE_CSV)
    if coverage.empty:
        raise FileNotFoundError(coverage_dir / COVERAGE_ROWS_CSV)
    stable = ambiguous[ambiguous["has_stable_route_evidence"].astype(bool)].copy()
    if stable.empty:
        raise ValueError("no stable ambiguous rows available for refinement")

    index_cols = [
        "panel_pair_id",
        "left_representative_candidate_index",
        "right_representative_candidate_index",
    ]
    stable = stable.merge(coverage[index_cols], on="panel_pair_id", how="left")
    endpoint_index, baseline_index = _load_cache_index(endpoint_cache_dir)

    rows: list[dict[str, Any]] = []
    cache_links: list[dict[str, Any]] = []
    for _, row in stable.iterrows():
        result, links = _analyze_pair(row, endpoint_cache_dir, endpoint_index, baseline_index)
        rows.append(result)
        cache_links.extend(links)

    refined = pd.DataFrame(rows).sort_values(
        ["identity_refinement_status", "distinct_threshold_margin", "panel_pair_id"],
        ascending=[True, True, True],
    )
    links_frame = pd.DataFrame(cache_links)
    summary = {
        "status": "stable_ambiguous_relation_refinement_prepared",
        "date": "2026-05-28",
        "stable_ambiguous_input_pair_count": int(len(stable)),
        "refined_pair_count": int(len(refined)),
        "identity_refinement_status_counts": refined[
            "identity_refinement_status"
        ].value_counts().to_dict(),
        "route_promotion_status_counts": refined[
            "route_promotion_status"
        ].value_counts().to_dict(),
        "route_promotion_eligible_count": int(
            refined["route_promotion_status"].eq(
                "eligible_for_route_gate_recheck_not_promoted_here"
            ).sum()
        ),
        "same_support_max": SAME_SUPPORT_MAX,
        "distinct_support_min": DISTINCT_SUPPORT_MIN,
        "near_same_margin": NEAR_SAME_MARGIN,
        "near_distinct_margin": NEAR_DISTINCT_MARGIN,
        "decision": (
            "Keep stable ambiguous rows blocked from wall promotion; use cached "
            "membership evidence to refine the basin relation rule first."
        ),
        "claim_boundary": (
            "Relation refinement only; no route execution, wall promotion, "
            "or basin evaluation is made."
        ),
    }

    _write_csv(refined, output_dir / REFINEMENT_ROWS_CSV)
    _write_csv(links_frame, output_dir / ENDPOINT_CACHE_LINKS_CSV)
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "script": _rel(Path(__file__)),
                "coverage_dir": _rel(coverage_dir),
                "endpoint_cache_dir": _rel(endpoint_cache_dir),
                "same_support_max": SAME_SUPPORT_MAX,
                "distinct_support_min": DISTINCT_SUPPORT_MIN,
                "near_same_margin": NEAR_SAME_MARGIN,
                "near_distinct_margin": NEAR_DISTINCT_MARGIN,
                "scope": "stable ambiguous relation refinement only; no route execution",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, refined)
    return {"output_dir": _rel(output_dir), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument("--endpoint-cache-dir", type=Path, default=DEFAULT_ENDPOINT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.coverage_dir, args.endpoint_cache_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
