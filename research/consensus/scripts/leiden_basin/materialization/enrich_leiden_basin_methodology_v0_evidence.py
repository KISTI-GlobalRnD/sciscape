#!/usr/bin/env python3
"""Enrich methodology-v0 panel pairs with endpoint/signature/cache evidence.

This is M2 in the Leiden basin methodology-v0 design. It joins endpoint
identity, support-local relation, distance, source-artifact, and cache
availability evidence onto the precommitted non-field34 panel. It does not
load memberships, run routes, promote walls, inspect quality/cost, or make a
directed-search claim.
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
DEFAULT_METHOD_DIR = BASE_RESULT_DIR / "leiden_basin_methodology_v0_20260529"
DEFAULT_CALIBRATION_DIR = BASE_RESULT_DIR / "leiden_basin_definition_calibration_20260528"
DEFAULT_CACHE_DIRS = [
    BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_endpoint_cache_20260528",
    BASE_RESULT_DIR / "leiden_basin_pending_membership_endpoint_cache_20260529",
]

PANEL_CSV = "precommitted_nonfield34_panel_v0.csv"
PAIR_CANDIDATES_CSV = "precommitted_nonfield34_pair_candidates_v0.csv"
ENDPOINT_ROWS_CSV = "endpoint_identity_rows.csv"
IDENTITY_PAIR_ROWS_CSV = "identity_pair_relation_rows.csv"

ENDPOINT_EVIDENCE_CSV = "methodology_v0_endpoint_evidence_rows.csv"
PAIR_EVIDENCE_CSV = "methodology_v0_pair_evidence_rows.csv"
SUMMARY_JSON = "methodology_v0_evidence_enrichment_summary.json"
REPORT_MD = "methodology_v0_evidence_enrichment_report.md"
CONFIG_JSON = "methodology_v0_evidence_enrichment_config.json"

CLAIM_BOUNDARY = (
    "Methodology-v0 evidence enrichment only; no route execution, "
    "wall-promotion change, basin-quality claim, cost claim, or "
    "directed-search claim."
)
QUALITY_COST_STATUS = "excluded_by_methodology_v0"
ROUTE_EXECUTION_STATUS = "not_executed_m2_evidence_enrichment_only"
WALL_PROMOTION_STATUS = "not_promoted_m2_evidence_enrichment_only"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve(path: Path) -> Path:
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


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return math.nan
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _load_cache_index(cache_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cache_dir in cache_dirs:
        resolved = _resolve(cache_dir)
        if not resolved.exists():
            continue
        for metadata_path in sorted(resolved.glob("*.metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if str(metadata.get("kind", "")) != "endpoint":
                continue
            candidate_index = _safe_int(metadata.get("candidate_index"))
            case_id = str(metadata.get("case_id", ""))
            if not case_id or candidate_index is None:
                continue
            cache_key = str(metadata.get("cache_key", metadata_path.stem))
            membership_path = metadata_path.with_name(cache_key + ".membership.npy")
            rows.append(
                {
                    "case_id": case_id,
                    "representative_candidate_index": candidate_index,
                    "cache_key": cache_key,
                    "cache_dir": _rel(resolved),
                    "membership_path": _rel(membership_path),
                    "metadata_path": _rel(metadata_path),
                    "membership_hash": str(metadata.get("membership_hash", "")),
                    "cached_support_node_count": _safe_int(metadata.get("support_node_count")),
                    "cache_resolution": metadata.get("resolution", ""),
                    "cache_randomness": metadata.get("randomness", ""),
                    "cache_seed": metadata.get("seed", ""),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "case_id",
                "representative_candidate_index",
                "cache_key",
                "cache_dir",
                "membership_path",
                "metadata_path",
                "membership_hash",
                "cached_support_node_count",
                "cache_resolution",
                "cache_randomness",
                "cache_seed",
            ]
        )
    frame = pd.DataFrame(rows)
    # Prefer the first cache directory in DEFAULT_CACHE_DIRS order by preserving
    # input order, then remove exact endpoint duplicates.
    return frame.drop_duplicates(
        subset=["case_id", "representative_candidate_index"],
        keep="first",
    ).reset_index(drop=True)


def _endpoint_lookup(endpoint_rows: pd.DataFrame) -> pd.DataFrame:
    rows = endpoint_rows.copy()
    rows["representative_candidate_index"] = rows["representative_candidate_index"].map(_safe_int)
    rows["candidate_index"] = rows["candidate_index"].map(_safe_int)
    rows["representative_candidate_index"] = rows["representative_candidate_index"].fillna(
        rows["candidate_index"]
    )
    rows = rows.sort_values(["case_id", "endpoint_identity_id", "candidate_index"])
    return rows.drop_duplicates(subset=["case_id", "endpoint_identity_id"], keep="first")


def _endpoint_evidence_grade(row: pd.Series) -> str:
    if _safe_text(row.get("cache_key")):
        return "full_membership_cache_available"
    if _safe_text(row.get("endpoint_signature")) and _safe_text(row.get("support_node_hash")):
        return "endpoint_signature_and_support_hash_available"
    if _safe_text(row.get("endpoint_signature")):
        return "endpoint_signature_only"
    return "endpoint_identity_reference_only"


def _endpoint_evidence_rows(
    *,
    pair_candidates: pd.DataFrame,
    endpoint_rows: pd.DataFrame,
    cache_index: pd.DataFrame,
) -> pd.DataFrame:
    endpoint_ids = pd.concat(
        [
            pair_candidates[
                ["case_id", "left_endpoint_identity_id"]
            ].rename(columns={"left_endpoint_identity_id": "endpoint_identity_id"}),
            pair_candidates[
                ["case_id", "right_endpoint_identity_id"]
            ].rename(columns={"right_endpoint_identity_id": "endpoint_identity_id"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    rows = endpoint_ids.merge(
        _endpoint_lookup(endpoint_rows),
        on=["case_id", "endpoint_identity_id"],
        how="left",
    )
    rows = rows.merge(
        cache_index,
        on=["case_id", "representative_candidate_index"],
        how="left",
    )
    rows["membership_cache_status"] = rows["cache_key"].map(
        lambda value: "cached_full_membership_metadata"
        if _safe_text(value)
        else "cache_not_found_in_known_endpoint_caches"
    )
    rows["endpoint_evidence_grade"] = rows.apply(_endpoint_evidence_grade, axis=1)
    rows["support_hash_cache_check"] = rows.apply(_support_hash_cache_check, axis=1)
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    cols = [
        "case_id",
        "field",
        "method",
        "candidate_budget",
        "endpoint_identity_id",
        "endpoint_signature",
        "endpoint_filter_status",
        "identity_member_count",
        "representative_candidate_index",
        "support_node_count",
        "support_node_hash",
        "sketch_node_hash",
        "sketch_sample_size",
        "source_artifact",
        "membership_cache_status",
        "cache_key",
        "membership_hash",
        "membership_path",
        "metadata_path",
        "cached_support_node_count",
        "support_hash_cache_check",
        "endpoint_evidence_grade",
        "quality_cost_status",
        "claim_boundary",
    ]
    for col in cols:
        if col not in rows:
            rows[col] = ""
    return rows[cols].sort_values(["case_id", "endpoint_identity_id"]).reset_index(drop=True)


def _support_hash_cache_check(row: pd.Series) -> str:
    cached_count = _safe_int(row.get("cached_support_node_count"))
    support_count = _safe_int(row.get("support_node_count"))
    if cached_count is None:
        return "not_checked_cache_missing"
    if support_count is None:
        return "not_checked_source_support_missing"
    if cached_count == support_count:
        return "cached_support_count_matches_source"
    return "cached_support_count_differs_from_source"


def _pair_evidence_grade(row: pd.Series) -> str:
    left = _safe_text(row.get("left_endpoint_evidence_grade"))
    right = _safe_text(row.get("right_endpoint_evidence_grade"))
    if left == "full_membership_cache_available" and right == "full_membership_cache_available":
        return "both_full_membership_caches_available"
    if "full_membership_cache_available" in {left, right}:
        return "partial_full_membership_cache_available"
    if left == "endpoint_signature_and_support_hash_available" and right == left:
        return "endpoint_signature_support_hash_pair"
    if left and right:
        return "endpoint_signature_pair"
    return "pair_evidence_incomplete"


def _support_distance_band(row: pd.Series) -> str:
    distance = _safe_float(row.get("support_distance_max"))
    if not math.isfinite(distance):
        return "support_distance_missing"
    if distance <= 0.5:
        return "same_zone"
    if distance >= 0.75:
        return "distinct_zone"
    return "boundary_band"


def _m3_readiness(row: pd.Series) -> str:
    if _safe_text(row.get("calibrated_relation")) != "distinct_support_local":
        return "not_ready_relation_not_accepted_distinct"
    if _safe_text(row.get("quality_cost_status")) != QUALITY_COST_STATUS:
        return "not_ready_quality_cost_leak"
    if _safe_text(row.get("pair_evidence_grade")) in {
        "both_full_membership_caches_available",
        "endpoint_signature_support_hash_pair",
        "partial_full_membership_cache_available",
    }:
        return "m3_schema_review_ready_missing_wall_evidence"
    return "m3_schema_review_hold_incomplete_endpoint_evidence"


def _pair_evidence_rows(
    *,
    pair_candidates: pd.DataFrame,
    endpoint_evidence: pd.DataFrame,
    identity_pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    endpoint_cols = [
        "case_id",
        "endpoint_identity_id",
        "endpoint_signature",
        "identity_member_count",
        "representative_candidate_index",
        "support_node_hash",
        "sketch_node_hash",
        "sketch_sample_size",
        "source_artifact",
        "membership_cache_status",
        "cache_key",
        "membership_hash",
        "membership_path",
        "metadata_path",
        "endpoint_evidence_grade",
        "support_hash_cache_check",
    ]
    left = endpoint_evidence[endpoint_cols].add_prefix("left_")
    right = endpoint_evidence[endpoint_cols].add_prefix("right_")
    rows = pair_candidates.merge(
        left,
        left_on=["case_id", "left_endpoint_identity_id"],
        right_on=["left_case_id", "left_endpoint_identity_id"],
        how="left",
    ).merge(
        right,
        left_on=["case_id", "right_endpoint_identity_id"],
        right_on=["right_case_id", "right_endpoint_identity_id"],
        how="left",
    )
    relation_cols = [
        "case_id",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "candidate_pair_count",
        "endpoint_distance_min",
        "endpoint_distance_max",
        "support_distance_min",
        "support_distance_max",
        "calibrated_relation",
        "relation_notes",
    ]
    relations = identity_pair_rows[relation_cols].rename(
        columns={
            "endpoint_distance_min": "identity_pair_endpoint_distance_min",
            "endpoint_distance_max": "identity_pair_endpoint_distance_max",
            "support_distance_min": "identity_pair_support_distance_min",
            "support_distance_max": "identity_pair_support_distance_max",
            "calibrated_relation": "identity_pair_calibrated_relation",
            "relation_notes": "identity_pair_relation_notes",
        }
    )
    rows = rows.merge(
        relations,
        on=["case_id", "left_endpoint_identity_id", "right_endpoint_identity_id"],
        how="left",
    )
    rows["support_distance_band_v0"] = rows.apply(_support_distance_band, axis=1)
    rows["pair_evidence_grade"] = rows.apply(_pair_evidence_grade, axis=1)
    rows["m2_evidence_status"] = "evidence_enriched_no_quality_cost"
    rows["m3_schema_review_status"] = rows.apply(_m3_readiness, axis=1)
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    cols = [
        "case_id",
        "field",
        "method",
        "candidate_budget",
        "panel_role",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "left_representative_candidate_index",
        "right_representative_candidate_index",
        "left_endpoint_evidence_grade",
        "right_endpoint_evidence_grade",
        "pair_evidence_grade",
        "left_membership_cache_status",
        "right_membership_cache_status",
        "left_membership_hash",
        "right_membership_hash",
        "left_membership_path",
        "right_membership_path",
        "left_support_hash_cache_check",
        "right_support_hash_cache_check",
        "endpoint_distance_min",
        "endpoint_distance_max",
        "support_distance_min",
        "support_distance_max",
        "identity_pair_endpoint_distance_min",
        "identity_pair_endpoint_distance_max",
        "identity_pair_support_distance_min",
        "identity_pair_support_distance_max",
        "calibrated_relation",
        "identity_pair_calibrated_relation",
        "support_distance_band_v0",
        "support_substance_class",
        "meaningful_basin_pair_status",
        "panel_pair_status",
        "wall_evidence_allowed",
        "m2_evidence_status",
        "m3_schema_review_status",
        "route_label_v0",
        "required_next_evidence",
        "quality_cost_status",
        "route_execution_status",
        "wall_promotion_status",
        "claim_boundary",
    ]
    for col in cols:
        if col not in rows:
            rows[col] = ""
    return rows[cols].sort_values(
        ["panel_role", "case_id", "pair_evidence_grade", "support_distance_max"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def _summary(
    *,
    endpoint_evidence: pd.DataFrame,
    pair_evidence: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    m3_ready = int(
        pair_evidence["m3_schema_review_status"]
        .eq("m3_schema_review_ready_missing_wall_evidence")
        .sum()
    )
    both_cached = int(
        pair_evidence["pair_evidence_grade"].eq("both_full_membership_caches_available").sum()
    )
    quality_cost_excluded = bool(pair_evidence["quality_cost_status"].eq(QUALITY_COST_STATUS).all())
    route_not_run = bool(pair_evidence["route_execution_status"].eq(ROUTE_EXECUTION_STATUS).all())
    wall_not_promoted = bool(pair_evidence["wall_promotion_status"].eq(WALL_PROMOTION_STATUS).all())
    return {
        "status": "methodology_v0_evidence_enrichment_prepared",
        "date": "2026-05-29",
        "script": _rel(Path(__file__).resolve()),
        "output_dir": _rel(output_dir),
        "endpoint_evidence_row_count": int(len(endpoint_evidence)),
        "pair_evidence_row_count": int(len(pair_evidence)),
        "endpoint_evidence_grade_counts": _count(endpoint_evidence, "endpoint_evidence_grade"),
        "endpoint_cache_status_counts": _count(endpoint_evidence, "membership_cache_status"),
        "pair_evidence_grade_counts": _count(pair_evidence, "pair_evidence_grade"),
        "m3_schema_review_status_counts": _count(pair_evidence, "m3_schema_review_status"),
        "m3_schema_review_ready_pair_count": m3_ready,
        "both_full_membership_cache_pair_count": both_cached,
        "quality_cost_excluded": quality_cost_excluded,
        "route_execution_not_run": route_not_run,
        "wall_promotion_not_run": wall_not_promoted,
        "decision": (
            "M2 evidence enrichment is complete: endpoint/signature/cache evidence is "
            "attached and quality/cost remain excluded. Pairs are ready for M3 schema "
            "review, not route execution."
        ),
        "next_step": (
            "Run M3 wall/pathway schema review on enriched pairs. Require direct-path, "
            "objective-debt, recovery, polish-reversion, or support-incompatibility "
            "evidence before any pathway probe."
        ),
        "paths": {
            "endpoint_evidence": _rel(output_dir / ENDPOINT_EVIDENCE_CSV),
            "pair_evidence": _rel(output_dir / PAIR_EVIDENCE_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
            "config": _rel(output_dir / CONFIG_JSON),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    path: Path,
    summary: dict[str, Any],
    endpoint_evidence: pd.DataFrame,
    pair_evidence: pd.DataFrame,
) -> None:
    lines = [
        "# Methodology v0 Evidence Enrichment",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact materializes M2 from the precommitted non-field34 panel.",
        "It joins endpoint identity, support-local distance, signature, source,",
        "and available cache metadata. It does not load memberships, run routes,",
        "promote wall claims, or inspect quality/cost.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "## Counts",
        "",
        f"- endpoint evidence rows: `{summary['endpoint_evidence_row_count']}`",
        f"- pair evidence rows: `{summary['pair_evidence_row_count']}`",
        f"- M3 schema-review ready pairs: `{summary['m3_schema_review_ready_pair_count']}`",
        f"- pairs with both full-membership caches available: "
        f"`{summary['both_full_membership_cache_pair_count']}`",
        "",
        "## Pair Evidence Grades",
        "",
        "| grade | rows |",
        "| --- | ---: |",
    ]
    for grade, count in summary["pair_evidence_grade_counts"].items():
        lines.append(f"| {grade} | {count} |")
    lines.extend(
        [
            "",
            "## M3 Readiness",
            "",
            "| status | rows |",
            "| --- | ---: |",
        ]
    )
    for status, count in summary["m3_schema_review_status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Endpoint Evidence Grades",
            "",
            "| grade | rows |",
            "| --- | ---: |",
        ]
    )
    for grade, count in summary["endpoint_evidence_grade_counts"].items():
        lines.append(f"| {grade} | {count} |")
    lines.extend(
        [
            "",
            "## No-Leak Checks",
            "",
            f"- quality/cost excluded: `{str(summary['quality_cost_excluded']).lower()}`",
            f"- route execution not run: `{str(summary['route_execution_not_run']).lower()}`",
            f"- wall promotion not run: `{str(summary['wall_promotion_not_run']).lower()}`",
            "",
            "## Next Step",
            "",
            str(summary["next_step"]),
            "",
            "Claim boundary: " + CLAIM_BOUNDARY,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    methodology_dir: Path,
    calibration_dir: Path,
    cache_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_candidates = _read_csv(methodology_dir / PAIR_CANDIDATES_CSV)
    endpoint_rows = _read_csv(calibration_dir / ENDPOINT_ROWS_CSV)
    identity_pair_rows = _read_csv(calibration_dir / IDENTITY_PAIR_ROWS_CSV)
    cache_index = _load_cache_index(cache_dirs)

    endpoint_evidence = _endpoint_evidence_rows(
        pair_candidates=pair_candidates,
        endpoint_rows=endpoint_rows,
        cache_index=cache_index,
    )
    pair_evidence = _pair_evidence_rows(
        pair_candidates=pair_candidates,
        endpoint_evidence=endpoint_evidence,
        identity_pair_rows=identity_pair_rows,
    )
    summary = _summary(
        endpoint_evidence=endpoint_evidence,
        pair_evidence=pair_evidence,
        output_dir=output_dir,
    )

    _write_csv(endpoint_evidence, output_dir / ENDPOINT_EVIDENCE_CSV)
    _write_csv(pair_evidence, output_dir / PAIR_EVIDENCE_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "methodology_dir": _rel(methodology_dir),
                "calibration_dir": _rel(calibration_dir),
                "cache_dirs": [_rel(_resolve(path)) for path in cache_dirs],
                "output_dir": _rel(output_dir),
                "quality_cost_status": QUALITY_COST_STATUS,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, endpoint_evidence, pair_evidence)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methodology-dir", type=Path, default=DEFAULT_METHOD_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--cache-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METHOD_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dirs = args.cache_dir if args.cache_dir is not None else DEFAULT_CACHE_DIRS
    summary = run(
        methodology_dir=args.methodology_dir,
        calibration_dir=args.calibration_dir,
        cache_dirs=cache_dirs,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
