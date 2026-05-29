#!/usr/bin/env python3
"""Materialize full memberships for pending Leiden basin relation checks.

This is cache/materialization-only evidence for Track C. It recreates baseline
and endpoint memberships for rows already marked
`pending_membership_relation_check`, then compares the resulting full changed
support against the candidate-row support evidence. It does not run direct
routes, promote wall claims, or inspect basin quality/cost as research evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT / "leiden_basin/transition_routes/route_wall"))
sys.path.insert(0, str(REPO_ROOT))

from collect_leiden_vanilla_reachability_sweep import hash_u32_sequence  # noqa: E402
from run_leiden_basin_transition_operator_pilot import (  # noqa: E402
    CandidateMembership,
    RecreatedMembership,
    _recreate_candidate,
    _run_leiden,
    changed_support_nodes,
    support_distance,
)
from run_leiden_basin_uniform_direct_pair_routes import (  # noqa: E402
    _cache_key,
    _cache_paths,
    _load_graph,
    _load_membership_cache,
    _membership_hash,
    _save_membership_cache,
)


BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_PENDING_REVIEW_DIR = (
    BASE_RESULT_DIR / "leiden_basin_pending_membership_relation_review_20260529"
)
DEFAULT_COVERAGE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_wall_panel_context_coverage_after_clean_distinct_route_gate_20260528"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_pending_membership_cache_materialization_20260529"
)
DEFAULT_CACHE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_pending_membership_endpoint_cache_20260529"
)
DEFAULT_GRAPH_ROOT = BASE_RESULT_DIR / "leiden_hysteresis_exception_detector_graphs_20260514"

PENDING_ROWS_CSV = "pending_membership_relation_review_rows.csv"
COVERAGE_ROWS_CSV = "wall_panel_context_coverage_rows.csv"

CACHE_ROWS_CSV = "pending_membership_cache_rows.csv"
RELATION_ROWS_CSV = "pending_membership_cache_relation_rows.csv"
SUMMARY_JSON = "pending_membership_cache_materialization_summary.json"
REPORT_MD = "pending_membership_cache_materialization_report.md"
CONFIG_JSON = "pending_membership_cache_materialization_config.json"
PROGRESS_JSONL = "pending_membership_cache_materialization_progress.jsonl"

BASELINE_ITERATIONS = 10
POLISH_ITERATIONS = 5
RESOLUTION = 0.01
RANDOMNESS = 0.01
PERTURB_SEED_OFFSET = 5000
SAME_SUPPORT_MAX = 0.5
DISTINCT_SUPPORT_MIN = 0.75
CLAIM_BOUNDARY = (
    "Pending-membership cache materialization only; no route execution, "
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


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _emit_progress(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"timestamp_utc": _now_utc(), **event}, sort_keys=True) + "\n"
        )


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _case_tail(case_id: str) -> str:
    for suffix in ("_budget12", "_budget15"):
        if case_id.endswith(suffix):
            return case_id[: -len(suffix)]
    return case_id


def _find_candidate_row(path: Path, case_id: str, candidate_index: int) -> pd.Series:
    frame = _read_csv(path)
    rows = frame[
        frame["case"].astype(str).str.endswith(_case_tail(case_id))
        & pd.to_numeric(frame["candidate_index"], errors="coerce").eq(candidate_index)
    ]
    if rows.empty:
        raise ValueError(f"missing candidate {candidate_index} for {case_id} in {path}")
    return rows.iloc[0]


def _graph_method_from_case(case_id: str) -> tuple[str, str]:
    tail = _case_tail(case_id)
    marker = "_gcc_emb_full_knn30_"
    if marker in tail:
        sample, graph_method = tail.split(marker, 1)
        return f"{sample}_gcc_emb_full_knn30", graph_method
    marker = "_all_edges_"
    if marker in tail:
        sample, graph_method = tail.split(marker, 1)
        return f"{sample}_all_edges", graph_method
    raise ValueError(f"cannot infer graph method from case_id={case_id!r}")


def _graph_key_for_row(row: pd.Series, coverage_row: pd.Series | None, graph_root: Path) -> str:
    if coverage_row is not None:
        graph_dir = str(coverage_row.get("runner_preflight_graph_dir", "")).strip()
        if graph_dir and graph_dir.lower() != "nan":
            return graph_dir
    sample, graph_method = _graph_method_from_case(str(row["case_id"]))
    inferred = graph_root / sample / graph_method
    if not inferred.exists():
        raise FileNotFoundError(f"cannot infer graph_dir for {row['panel_pair_id']}: {inferred}")
    return _rel(inferred)


def _support_hash(nodes: np.ndarray) -> str:
    return hash_u32_sequence(np.asarray(nodes, dtype=np.uint32))


def _classification(distance: float) -> str:
    if distance <= SAME_SUPPORT_MAX:
        return "same_support_local"
    if distance >= DISTINCT_SUPPORT_MIN:
        return "distinct_support_local"
    return "boundary_review_ambiguous_support_local"


def _load_or_run_baseline(
    *,
    graph: Any,
    graph_key: str,
    cache_dir: Path,
    case_id: str,
    seed: int,
    reuse_cache: bool,
    cache_rows: list[dict[str, Any]],
) -> tuple[RecreatedMembership, str, str]:
    payload = {
        "kind": "baseline",
        "graph_dir": graph_key,
        "case_id": case_id,
        "seed": int(seed),
        "baseline_iterations": BASELINE_ITERATIONS,
        "resolution": RESOLUTION,
        "randomness": RANDOMNESS,
    }
    key = _cache_key(payload)
    cached = _load_membership_cache(cache_dir=cache_dir, key=key) if reuse_cache else None
    if cached is not None:
        membership, metadata = cached
        quality = _safe_float(metadata.get("objective_value"))
        if not math.isfinite(quality):
            quality = float(graph.cpm_quality(membership, resolution=RESOLUTION))
        status = "hit"
        recreated = RecreatedMembership(membership=membership, quality=quality, elapsed_sec=0.0)
    else:
        recreated = _run_leiden(
            graph,
            resolution=RESOLUTION,
            seed=seed,
            n_iterations=BASELINE_ITERATIONS,
            randomness=RANDOMNESS,
        )
        _save_membership_cache(
            cache_dir=cache_dir,
            key=key,
            membership=recreated.membership,
            metadata={
                **payload,
                "cache_key": key,
                "objective_value": recreated.quality,
                "membership_hash": _membership_hash(recreated.membership),
                "created_at_utc": _now_utc(),
            },
        )
        status = "miss_materialized"
    membership_path, metadata_path = _cache_paths(cache_dir, key)
    cache_rows.append(
        {
            "cache_kind": "baseline",
            "cache_status": status,
            "cache_key": key,
            "case_id": case_id,
            "candidate_index": "",
            "membership_hash": _membership_hash(recreated.membership),
            "membership_path": _rel(membership_path),
            "metadata_path": _rel(metadata_path),
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    return recreated, key, status


def _load_or_run_endpoint(
    *,
    graph: Any,
    arrays: Any,
    node_weights: np.ndarray,
    graph_key: str,
    cache_dir: Path,
    case_id: str,
    baseline: RecreatedMembership,
    baseline_cache_key: str,
    row: pd.Series,
    reuse_cache: bool,
    cache_rows: list[dict[str, Any]],
) -> CandidateMembership:
    candidate_index = int(row["candidate_index"])
    payload = {
        "kind": "endpoint",
        "graph_dir": graph_key,
        "case_id": case_id,
        "candidate_index": candidate_index,
        "seed": int(row.get("seed", 0)),
        "source_cluster": int(row["source_cluster"]),
        "target_cluster": int(row["target_cluster"]),
        "baseline_cache_key": baseline_cache_key,
        "polish_iterations": POLISH_ITERATIONS,
        "resolution": RESOLUTION,
        "randomness": RANDOMNESS,
        "perturb_seed_offset": PERTURB_SEED_OFFSET,
    }
    key = _cache_key(payload)
    cached = _load_membership_cache(cache_dir=cache_dir, key=key) if reuse_cache else None
    if cached is not None:
        membership, metadata = cached
        quality = _safe_float(metadata.get("objective_value"))
        if not math.isfinite(quality):
            quality = float(graph.cpm_quality(membership, resolution=RESOLUTION))
        recreated = RecreatedMembership(membership=membership, quality=quality, elapsed_sec=0.0)
        support_nodes = changed_support_nodes(baseline.membership, membership)
        candidate = CandidateMembership(
            recreated=recreated,
            row=row,
            group_nodes=np.asarray([], dtype=np.uint32),
            support_nodes=support_nodes,
        )
        status = "hit"
    else:
        candidate = _recreate_candidate(
            graph=graph,
            arrays=arrays,
            node_weights=node_weights,
            baseline_membership=baseline.membership,
            baseline_quality=baseline.quality,
            row=row,
            resolution=RESOLUTION,
            randomness=RANDOMNESS,
            perturb_seed_offset=PERTURB_SEED_OFFSET,
            polish_iterations=POLISH_ITERATIONS,
        )
        _save_membership_cache(
            cache_dir=cache_dir,
            key=key,
            membership=candidate.recreated.membership,
            metadata={
                **payload,
                "cache_key": key,
                "objective_value": candidate.recreated.quality,
                "membership_hash": _membership_hash(candidate.recreated.membership),
                "support_node_count": int(candidate.support_nodes.size),
                "created_at_utc": _now_utc(),
            },
        )
        status = "miss_materialized"
    expected_hash = str(row.get("p5_basin_changed_support_node_hash", ""))
    computed_hash = _support_hash(candidate.support_nodes)
    expected_count = _safe_int(row.get("p5_basin_changed_support_node_count"), -1)
    membership_path, metadata_path = _cache_paths(cache_dir, key)
    cache_rows.append(
        {
            "cache_kind": "endpoint",
            "cache_status": status,
            "cache_key": key,
            "case_id": case_id,
            "candidate_index": candidate_index,
            "membership_hash": _membership_hash(candidate.recreated.membership),
            "computed_support_node_count": int(candidate.support_nodes.size),
            "expected_support_node_count": expected_count,
            "support_node_count_match": int(candidate.support_nodes.size) == expected_count,
            "computed_support_node_hash": computed_hash,
            "expected_support_node_hash": expected_hash,
            "support_node_hash_match": computed_hash == expected_hash,
            "membership_path": _rel(membership_path),
            "metadata_path": _rel(metadata_path),
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    return candidate


def run(
    *,
    pending_review_dir: Path,
    coverage_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    graph_root: Path,
    reuse_cache: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_JSONL
    if progress_path.exists():
        progress_path.unlink()

    pending = _read_csv(pending_review_dir / PENDING_ROWS_CSV)
    coverage = _read_csv(coverage_dir / COVERAGE_ROWS_CSV)
    coverage_by_pair = {str(row["panel_pair_id"]): row for _, row in coverage.iterrows()}
    pending = pending[
        pending["relation_queue_status"].astype(str).eq("pending_membership_relation_check")
    ].copy()
    if pending.empty:
        raise ValueError("no pending membership rows found")

    graph_cache: dict[str, tuple[Any, np.ndarray, Any]] = {}
    baseline_cache: dict[str, tuple[RecreatedMembership, str]] = {}
    cache_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for pair_number, (_, row) in enumerate(pending.iterrows(), start=1):
        panel_pair_id = str(row["panel_pair_id"])
        case_id = str(row["case_id"])
        _emit_progress(progress_path, {"event": "pair_start", "panel_pair_id": panel_pair_id})
        try:
            coverage_row = coverage_by_pair.get(panel_pair_id)
            graph_key = _graph_key_for_row(row, coverage_row, graph_root)
            graph_dir = _resolve(graph_key)
            if graph_key not in graph_cache:
                graph_cache[graph_key] = _load_graph(graph_dir)
            graph, node_weights, arrays = graph_cache[graph_key]

            left_index = int(row["left_candidate_index"])
            right_index = int(row["right_candidate_index"])
            left_source = _resolve(
                coverage_row["left_endpoint_source_artifact"] if coverage_row is not None else ""
            )
            right_source = _resolve(
                coverage_row["right_endpoint_source_artifact"] if coverage_row is not None else ""
            )
            left_row = _find_candidate_row(left_source, case_id, left_index)
            right_row = _find_candidate_row(right_source, case_id, right_index)
            seed = int(left_row.get("seed", 0))
            baseline_key = f"{graph_key}|{case_id}|seed={seed}"
            if baseline_key not in baseline_cache:
                baseline, baseline_cache_key, _status = _load_or_run_baseline(
                    graph=graph,
                    graph_key=graph_key,
                    cache_dir=cache_dir,
                    case_id=case_id,
                    seed=seed,
                    reuse_cache=reuse_cache,
                    cache_rows=cache_rows,
                )
                baseline_cache[baseline_key] = (baseline, baseline_cache_key)
            baseline, baseline_cache_key = baseline_cache[baseline_key]

            left = _load_or_run_endpoint(
                graph=graph,
                arrays=arrays,
                node_weights=node_weights,
                graph_key=graph_key,
                cache_dir=cache_dir,
                case_id=case_id,
                baseline=baseline,
                baseline_cache_key=baseline_cache_key,
                row=left_row,
                reuse_cache=reuse_cache,
                cache_rows=cache_rows,
            )
            right = _load_or_run_endpoint(
                graph=graph,
                arrays=arrays,
                node_weights=node_weights,
                graph_key=graph_key,
                cache_dir=cache_dir,
                case_id=case_id,
                baseline=baseline,
                baseline_cache_key=baseline_cache_key,
                row=right_row,
                reuse_cache=reuse_cache,
                cache_rows=cache_rows,
            )
            distance, intersection, union = support_distance(left.support_nodes, right.support_nodes)
            left_hash = _membership_hash(left.recreated.membership)
            right_hash = _membership_hash(right.recreated.membership)
            relation_rows.append(
                {
                    "panel_pair_id": panel_pair_id,
                    "case_id": case_id,
                    "graph_dir": graph_key,
                    "left_candidate_index": left_index,
                    "right_candidate_index": right_index,
                    "left_membership_hash": left_hash,
                    "right_membership_hash": right_hash,
                    "membership_hash_match": left_hash == right_hash,
                    "left_support_node_count_exact": int(left.support_nodes.size),
                    "right_support_node_count_exact": int(right.support_nodes.size),
                    "exact_support_intersection": int(intersection),
                    "exact_support_union": int(union),
                    "exact_support_distance": float(distance),
                    "previous_support_distance": row.get("exact_support_distance_from_nodes", ""),
                    "support_distance_delta_from_previous": (
                        float(distance) - _safe_float(row.get("exact_support_distance_from_nodes"))
                    ),
                    "current_hard_gate_classification": _classification(float(distance)),
                    "previous_hard_gate_classification": row.get(
                        "current_hard_gate_classification",
                        "",
                    ),
                    "left_support_hash_match": _support_hash(left.support_nodes)
                    == str(left_row.get("p5_basin_changed_support_node_hash", "")),
                    "right_support_hash_match": _support_hash(right.support_nodes)
                    == str(right_row.get("p5_basin_changed_support_node_hash", "")),
                    "wall_promotion_status_after_materialization": "no_wall_promotion",
                    "route_execution_status_after_materialization": "not_recommended",
                    "evidence_grade": "materialized_full_membership_exact_support",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
            _emit_progress(
                progress_path,
                {
                    "event": "pair_done",
                    "panel_pair_id": panel_pair_id,
                    "exact_support_distance": float(distance),
                    "classification": _classification(float(distance)),
                },
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-pair audit rows.
            message = f"{type(exc).__name__}: {exc}"
            errors.append({"panel_pair_id": panel_pair_id, "error": message})
            _emit_progress(
                progress_path,
                {"event": "pair_error", "panel_pair_id": panel_pair_id, "error": message},
            )

    _write_csv(output_dir / CACHE_ROWS_CSV, cache_rows)
    _write_csv(output_dir / RELATION_ROWS_CSV, relation_rows)
    relation_frame = pd.DataFrame(relation_rows)
    cache_frame = pd.DataFrame(cache_rows)
    summary = {
        "status": "completed_with_errors" if errors else "completed",
        "date": "2026-05-29",
        "script": "research/consensus/scripts/materialize_leiden_basin_pending_membership_cache.py",
        "pending_review_dir": _rel(pending_review_dir),
        "coverage_dir": _rel(coverage_dir),
        "output_dir": _rel(output_dir),
        "cache_dir": _rel(cache_dir),
        "graph_root": _rel(graph_root),
        "pending_pair_count": int(len(pending)),
        "materialized_relation_pair_count": int(len(relation_rows)),
        "cache_row_count": int(len(cache_rows)),
        "cache_status_counts": (
            {}
            if cache_frame.empty
            else {
                str(k): int(v)
                for k, v in cache_frame["cache_status"].value_counts().to_dict().items()
            }
        ),
        "hard_gate_classification_counts": (
            {}
            if relation_frame.empty
            else {
                str(k): int(v)
                for k, v in relation_frame["current_hard_gate_classification"]
                .value_counts()
                .to_dict()
                .items()
            }
        ),
        "support_hash_mismatch_count": (
            0
            if cache_frame.empty or "support_node_hash_match" not in cache_frame
            else int(cache_frame["support_node_hash_match"].eq(False).sum())
        ),
        "promoted_wall_claim_count": (
            0
            if relation_frame.empty
            else int(
                relation_frame["wall_promotion_status_after_materialization"]
                .ne("no_wall_promotion")
                .sum()
            )
        ),
        "immediate_route_execution_count": (
            0
            if relation_frame.empty
            else int(
                relation_frame["route_execution_status_after_materialization"].eq("ready").sum()
            )
        ),
        "errors": errors,
        "paths": {
            "cache_rows": _rel(output_dir / CACHE_ROWS_CSV),
            "relation_rows": _rel(output_dir / RELATION_ROWS_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
            "progress": _rel(progress_path),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "pending_review_dir": _rel(pending_review_dir),
                "coverage_dir": _rel(coverage_dir),
                "output_dir": _rel(output_dir),
                "cache_dir": _rel(cache_dir),
                "graph_root": _rel(graph_root),
                "reuse_cache": bool(reuse_cache),
                "baseline_iterations": BASELINE_ITERATIONS,
                "polish_iterations": POLISH_ITERATIONS,
                "resolution": RESOLUTION,
                "randomness": RANDOMNESS,
                "perturb_seed_offset": PERTURB_SEED_OFFSET,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, relation_rows)
    return summary


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Pending-Membership Cache Materialization",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact materializes full baseline and endpoint memberships for the",
        "pending-membership relation checks only. It does not run direct routes,",
        "promote wall claims, or inspect basin quality/cost.",
        "",
        "## Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- pending pairs: `{summary['pending_pair_count']}`",
        f"- materialized relation pairs: `{summary['materialized_relation_pair_count']}`",
        f"- cache status counts: `{summary['cache_status_counts']}`",
        f"- hard-gate counts: `{summary['hard_gate_classification_counts']}`",
        f"- support hash mismatches: `{summary['support_hash_mismatch_count']}`",
        f"- promoted wall claims: `{summary['promoted_wall_claim_count']}`",
        f"- immediate route-execution rows: `{summary['immediate_route_execution_count']}`",
        "",
        "## Relation Rows",
        "",
        "| panel_pair_id | exact_support_distance | hard_gate | left_hash_match | right_hash_match |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("panel_pair_id", "")),
                    str(row.get("exact_support_distance", "")),
                    str(row.get("current_hard_gate_classification", "")),
                    str(row.get("left_support_hash_match", "")),
                    str(row.get("right_support_hash_match", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "Claim boundary: " + CLAIM_BOUNDARY, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending-review-dir", type=Path, default=DEFAULT_PENDING_REVIEW_DIR)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument(
        "--no-reuse-cache",
        action="store_true",
        help="Ignore existing cache files and materialize memberships again.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(
        pending_review_dir=args.pending_review_dir,
        coverage_dir=args.coverage_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        graph_root=args.graph_root,
        reuse_cache=not args.no_reuse_cache,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
