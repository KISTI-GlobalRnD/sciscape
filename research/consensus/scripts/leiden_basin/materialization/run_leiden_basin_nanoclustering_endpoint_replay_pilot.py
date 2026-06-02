#!/usr/bin/env python3
"""Run a bounded NanoClustering endpoint-replay pilot.

This is an execution smoke/pilot for Track C, not a method-success claim.  It
uses the readiness artifact to find branch-specific raw graph inputs, starts
from the frozen source/dominant seed0 endpoint partition, runs the same
Leiden -> min-nano postprocess -> min-doc postprocess sequence used by the
NanoClustering seed sweep, and scores the terminal partition against the
frozen endpoint-family target handles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sciscape.clustering.leiden_rust import RustLeidenGraph, build_leiden_graph


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_READINESS_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_endpoint_replay_readiness_20260601"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_endpoint_replay_pilot_smoke_20260601"
)

ATTEMPT_PLAN_ROWS_CSV = "nanoclustering_endpoint_replay_readiness_attempt_plan_rows.csv"
ENDPOINT_TARGET_ROWS_CSV = (
    "nanoclustering_endpoint_replay_readiness_endpoint_target_rows.csv"
)
GRAPH_INPUT_ROWS_CSV = "nanoclustering_endpoint_replay_readiness_graph_input_rows.csv"
RUNNER_CONFIG_TEMPLATE_JSON = (
    "nanoclustering_endpoint_replay_readiness_runner_config_template.json"
)

PILOT_ATTEMPT_ROWS_CSV = "nanoclustering_endpoint_replay_pilot_attempt_rows.csv"
PILOT_TARGET_SCORE_ROWS_CSV = (
    "nanoclustering_endpoint_replay_pilot_target_score_rows.csv"
)
PILOT_CONFIG_JSON = "nanoclustering_endpoint_replay_pilot_config.json"
PILOT_SUMMARY_JSON = "nanoclustering_endpoint_replay_pilot_summary.json"
PILOT_REPORT_MD = "nanoclustering_endpoint_replay_pilot_report.md"

CLAIM_BOUNDARY = (
    "NanoClustering endpoint-replay pilot only; executes a bounded strict-core "
    "smoke run from frozen source endpoint partitions and scores terminal "
    "partitions against frozen endpoint-family target handles. It does not "
    "execute route/pathway intervention, promote walls, inspect quality/cost "
    "as success, or claim real-data method success."
)
REPLAY_EXECUTION_STATUS = "executed_endpoint_replay_pilot_smoke"
ROUTE_EXECUTION_STATUS = "not_executed_no_route_trace"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
QUALITY_COST_STATUS = "excluded_pilot_quality_cost_not_success"


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
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_csv_list(value: str | None, *, cast: Any = str) -> tuple[Any, ...]:
    if value is None or not str(value).strip():
        return ()
    return tuple(cast(part.strip()) for part in str(value).split(",") if part.strip())


def _compact_membership(membership: np.ndarray) -> np.ndarray:
    labels, _ = pd.factorize(np.asarray(membership), sort=False)
    return np.asarray(labels, dtype=np.uint64)


def _partition_hash(membership: np.ndarray) -> str:
    canonical = _compact_membership(membership)
    return hashlib.blake2b(
        np.ascontiguousarray(canonical, dtype=np.uint64).tobytes(),
        digest_size=16,
    ).hexdigest()


def _load_manifest(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_parquet(
        path,
        columns=["node_idx", "original_cluster_id", "doc_count"],
    ).sort_values("node_idx", kind="mergesort")
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["node_idx"].to_numpy(dtype=np.int64), expected):
        raise ValueError(f"node_idx is not dense and sorted in {path}")
    weights = frame["doc_count"].to_numpy(dtype=np.float64)
    return frame, weights


def _load_membership(
    path: Path,
    *,
    label_col: str = "candidate_micro_id",
) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=["node_idx", "original_cluster_id", "doc_count", label_col],
    ).sort_values("node_idx", kind="mergesort")


def _membership_array(path: Path, *, label_col: str = "candidate_micro_id") -> np.ndarray:
    frame = _load_membership(path, label_col=label_col)
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["node_idx"].to_numpy(dtype=np.int64), expected):
        raise ValueError(f"node_idx is not dense and sorted in {path}")
    return _compact_membership(frame[label_col].to_numpy(dtype=np.int64))


def _cluster_mask(
    path: Path,
    *,
    cluster_id: int,
    label_col: str = "candidate_micro_id",
) -> np.ndarray:
    frame = _load_membership(path, label_col=label_col)
    labels = frame[label_col].to_numpy(dtype=np.int64)
    return labels == int(cluster_id)


def _safe_metric(metric: Any, left: np.ndarray, right: np.ndarray) -> float:
    try:
        return float(metric(left, right))
    except Exception:
        return float("nan")


def _load_graph(
    graph_row: pd.Series,
    *,
    manifest_cache: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> tuple[RustLeidenGraph, pd.DataFrame, np.ndarray, float]:
    manifest_path = Path(str(graph_row["runtime_node_manifest_path"]))
    edge_path = Path(str(graph_row["runtime_int_edges_path"]))
    key = str(manifest_path)
    if key not in manifest_cache:
        manifest_cache[key] = _load_manifest(manifest_path)
    manifest, node_weights = manifest_cache[key]
    start = time.perf_counter()
    graph = build_leiden_graph(
        edge_path=edge_path,
        n_nodes=len(manifest),
        node_weights=node_weights,
    )
    return graph, manifest, node_weights, time.perf_counter() - start


def _select_attempts(
    attempt_plan: pd.DataFrame,
    *,
    analysis_tier: str,
    role_sides: tuple[str, ...],
    method_seeds: tuple[int, ...],
    case_ranks: tuple[int, ...],
    max_cases: int,
) -> pd.DataFrame:
    selected = attempt_plan[
        attempt_plan["analysis_tier"].astype(str).eq(analysis_tier)
        & attempt_plan["attempt_execution_status"].astype(str).eq("ready_to_execute")
    ].copy()
    if role_sides:
        selected = selected[selected["role_side"].astype(str).isin(role_sides)]
    if method_seeds:
        selected = selected[selected["method_seed"].astype(int).isin(method_seeds)]
    if case_ranks:
        selected = selected[selected["panel_case_rank"].astype(int).isin(case_ranks)]
    selected = selected.sort_values(
        ["panel_case_rank", "method_seed", "role_side"], kind="mergesort"
    )
    if max_cases > 0:
        keep_cases = (
            selected[["panel_case_id", "panel_case_rank"]]
            .drop_duplicates()
            .sort_values("panel_case_rank", kind="mergesort")
            .head(max_cases)["panel_case_id"]
            .astype(str)
            .tolist()
        )
        selected = selected[selected["panel_case_id"].astype(str).isin(keep_cases)]
    return selected.reset_index(drop=True)


def _source_rows_for_signature(
    target_rows: pd.DataFrame,
    endpoint_signature_id: str,
) -> pd.DataFrame:
    signature_rows = target_rows[
        target_rows["endpoint_signature_id"].astype(str).eq(endpoint_signature_id)
    ].copy()
    source = signature_rows[
        signature_rows["target_handle_role"].astype(str).eq(
            "dominant_host_context_member"
        )
    ]
    if source.empty:
        return signature_rows.head(1)
    return source.sort_values(["seed", "endpoint_handle_id"], kind="mergesort")


def _target_rows_for_signature(
    target_rows: pd.DataFrame,
    endpoint_signature_id: str,
) -> pd.DataFrame:
    rows = target_rows[
        target_rows["endpoint_signature_id"].astype(str).eq(endpoint_signature_id)
    ].copy()
    rows = rows[
        rows["target_handle_role"].astype(str).eq("top1_endpoint_target_member")
        & rows["membership_path_exists"].astype(bool)
        & rows["cluster_label_present"].astype(bool)
    ]
    return rows.sort_values(["seed", "endpoint_handle_id"], kind="mergesort")


def _score_target_handles(
    *,
    attempt: pd.Series,
    final: np.ndarray,
    initial: np.ndarray,
    node_weights: np.ndarray,
    target_rows: pd.DataFrame,
    membership_cache: dict[str, np.ndarray],
    mask_cache: dict[tuple[str, int], np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_labels = np.asarray(final, dtype=np.int64)
    initial_labels = np.asarray(initial, dtype=np.int64)
    final_counts = np.bincount(final_labels)
    final_weights = np.bincount(final_labels, weights=node_weights)
    initial_nmi = _safe_metric(normalized_mutual_info_score, initial_labels, final_labels)
    initial_ari = _safe_metric(adjusted_rand_score, initial_labels, final_labels)
    initial_exact = _partition_hash(initial) == _partition_hash(final)

    score_rows: list[dict[str, Any]] = []
    for target in target_rows.itertuples(index=False):
        row = target._asdict()
        path = Path(str(row["membership_path"]))
        label_col = str(row["label_cols"]).split(";")[0] or "candidate_micro_id"
        mem_key = str(path)
        if mem_key not in membership_cache:
            membership_cache[mem_key] = _membership_array(path, label_col=label_col)
        target_partition = membership_cache[mem_key]
        mask_key = (mem_key, int(row["cluster_id"]))
        if mask_key not in mask_cache:
            mask_cache[mask_key] = _cluster_mask(
                path,
                cluster_id=int(row["cluster_id"]),
                label_col=label_col,
            )
        target_mask = mask_cache[mask_key]
        target_units = int(target_mask.sum())
        target_weight = float(node_weights[target_mask].sum())
        if target_units == 0:
            best_label = -1
            overlap_units = 0
            overlap_weight = 0.0
            unit_recall = unit_precision = unit_f1 = 0.0
            weight_recall = weight_precision = weight_f1 = 0.0
        else:
            selected_final = final_labels[target_mask]
            overlap_counts = np.bincount(selected_final, minlength=len(final_counts))
            best_label = int(overlap_counts.argmax())
            overlap_units = int(overlap_counts[best_label])
            overlap_weight_by_label = np.bincount(
                selected_final,
                weights=node_weights[target_mask],
                minlength=len(final_weights),
            )
            overlap_weight = float(overlap_weight_by_label[best_label])
            unit_recall = overlap_units / target_units
            unit_precision = (
                overlap_units / float(final_counts[best_label])
                if final_counts[best_label]
                else 0.0
            )
            unit_f1 = (
                2.0 * unit_precision * unit_recall / (unit_precision + unit_recall)
                if unit_precision + unit_recall > 0
                else 0.0
            )
            weight_recall = overlap_weight / target_weight if target_weight else 0.0
            weight_precision = (
                overlap_weight / float(final_weights[best_label])
                if final_weights[best_label]
                else 0.0
            )
            weight_f1 = (
                2.0 * weight_precision * weight_recall / (weight_precision + weight_recall)
                if weight_precision + weight_recall > 0
                else 0.0
            )
        target_labels = np.asarray(target_partition, dtype=np.int64)
        partition_nmi = _safe_metric(
            normalized_mutual_info_score, target_labels, final_labels
        )
        partition_ari = _safe_metric(adjusted_rand_score, target_labels, final_labels)
        score_rows.append(
            {
                "attempt_id": attempt["attempt_id"],
                "panel_case_id": attempt["panel_case_id"],
                "panel_case_rank": int(attempt["panel_case_rank"]),
                "role_side": attempt["role_side"],
                "branch": attempt["branch"],
                "method_seed": int(attempt["method_seed"]),
                "endpoint_signature_id": attempt["target_endpoint_signature_id"],
                "endpoint_handle_id": row["endpoint_handle_id"],
                "target_run_id": row["run_id"],
                "target_seed": int(row["seed"]),
                "target_cluster_id": int(row["cluster_id"]),
                "target_unit_count": target_units,
                "target_weight_sum": target_weight,
                "best_terminal_cluster_id": best_label,
                "overlap_unit_count": overlap_units,
                "overlap_weight_sum": overlap_weight,
                "cluster_unit_recall": unit_recall,
                "cluster_unit_precision": unit_precision,
                "cluster_unit_f1": unit_f1,
                "cluster_weight_recall": weight_recall,
                "cluster_weight_precision": weight_precision,
                "cluster_weight_f1": weight_f1,
                "terminal_vs_target_partition_nmi": partition_nmi,
                "terminal_vs_target_partition_ari": partition_ari,
                "terminal_vs_initial_partition_nmi": initial_nmi,
                "terminal_vs_initial_partition_ari": initial_ari,
                "terminal_equals_initial_partition": initial_exact,
                "replay_execution_status": REPLAY_EXECUTION_STATUS,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    if score_rows:
        best = max(score_rows, key=lambda row: (row["cluster_weight_f1"], row["cluster_unit_f1"]))
    else:
        best = {
            "endpoint_handle_id": "",
            "target_run_id": "",
            "target_seed": None,
            "target_cluster_id": None,
            "cluster_unit_recall": None,
            "cluster_unit_precision": None,
            "cluster_unit_f1": None,
            "cluster_weight_recall": None,
            "cluster_weight_precision": None,
            "cluster_weight_f1": None,
            "terminal_vs_target_partition_nmi": None,
            "terminal_vs_target_partition_ari": None,
        }
    best_summary = {
        "terminal_vs_initial_partition_nmi": initial_nmi,
        "terminal_vs_initial_partition_ari": initial_ari,
        "terminal_equals_initial_partition": initial_exact,
        "best_target_handle_id": best.get("endpoint_handle_id", ""),
        "best_target_run_id": best.get("target_run_id", ""),
        "best_target_seed": best.get("target_seed"),
        "best_target_cluster_id": best.get("target_cluster_id"),
        "best_cluster_unit_recall": best.get("cluster_unit_recall"),
        "best_cluster_unit_precision": best.get("cluster_unit_precision"),
        "best_cluster_unit_f1": best.get("cluster_unit_f1"),
        "best_cluster_weight_recall": best.get("cluster_weight_recall"),
        "best_cluster_weight_precision": best.get("cluster_weight_precision"),
        "best_cluster_weight_f1": best.get("cluster_weight_f1"),
        "best_terminal_vs_target_partition_nmi": best.get(
            "terminal_vs_target_partition_nmi"
        ),
        "best_terminal_vs_target_partition_ari": best.get(
            "terminal_vs_target_partition_ari"
        ),
    }
    return score_rows, best_summary


def _run_attempt(
    *,
    attempt: pd.Series,
    graph: RustLeidenGraph,
    node_weights: np.ndarray,
    target_rows: pd.DataFrame,
    gamma: float,
    min_nano: int,
    min_docs: float,
    n_iterations: int,
    post_iterations: int,
    post_max_rounds: int,
    membership_cache: dict[str, np.ndarray],
    mask_cache: dict[tuple[str, int], np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    signature_id = str(attempt["target_endpoint_signature_id"])
    source_rows = _source_rows_for_signature(target_rows, signature_id)
    if source_rows.empty:
        raise ValueError(f"missing source rows for {signature_id}")
    source_paths = source_rows["membership_path"].astype(str).drop_duplicates().tolist()
    source = source_rows.iloc[0]
    initial_path = Path(str(source["membership_path"]))
    label_col = str(source["label_cols"]).split(";")[0] or "candidate_micro_id"
    initial = _membership_array(initial_path, label_col=label_col)

    method_seed = int(attempt["method_seed"])
    t0 = time.perf_counter()
    raw = graph.run_leiden(
        resolution=float(gamma),
        n_iterations=int(n_iterations),
        seed=method_seed,
        initial_membership=np.asarray(initial, dtype=np.uint64),
    )
    raw_seconds = time.perf_counter() - t0
    raw_mem = _compact_membership(raw.membership)

    t1 = time.perf_counter()
    post_nano = graph.postprocess_small_clusters(
        resolution=float(gamma),
        min_size=int(min_nano),
        min_weight=0.0,
        membership=np.asarray(raw_mem, dtype=np.uint64),
        seed=method_seed,
        n_iterations=int(post_iterations),
        max_rounds=int(post_max_rounds),
    )
    post_nano_seconds = time.perf_counter() - t1
    post_nano_mem = _compact_membership(post_nano.membership)

    t2 = time.perf_counter()
    post_doc = graph.postprocess_small_clusters(
        resolution=float(gamma),
        min_size=0,
        min_weight=float(min_docs),
        membership=np.asarray(post_nano_mem, dtype=np.uint64),
        seed=method_seed,
        n_iterations=int(post_iterations),
        max_rounds=int(post_max_rounds),
    )
    post_doc_seconds = time.perf_counter() - t2
    final = _compact_membership(post_doc.membership)
    final_quality = graph.cpm_quality(
        np.asarray(final, dtype=np.uint64),
        resolution=float(gamma),
    )
    raw_quality_weighted = graph.cpm_quality(
        np.asarray(raw_mem, dtype=np.uint64),
        resolution=float(gamma),
    )

    family_targets = _target_rows_for_signature(target_rows, signature_id)
    score_rows, best_summary = _score_target_handles(
        attempt=attempt,
        final=final,
        initial=initial,
        node_weights=node_weights,
        target_rows=family_targets,
        membership_cache=membership_cache,
        mask_cache=mask_cache,
    )
    attempt_row = {
        "attempt_id": attempt["attempt_id"],
        "panel_case_id": attempt["panel_case_id"],
        "panel_case_rank": int(attempt["panel_case_rank"]),
        "analysis_tier": attempt["analysis_tier"],
        "role_id": attempt["role_id"],
        "role_side": attempt["role_side"],
        "primitive_id": attempt["primitive_id"],
        "branch": attempt["branch"],
        "method_seed": method_seed,
        "target_endpoint_signature_id": signature_id,
        "initial_source_handle_id": source["endpoint_handle_id"],
        "initial_source_run_id": source["run_id"],
        "initial_source_seed": int(source["seed"]),
        "initial_source_cluster_id": int(source["cluster_id"]),
        "initial_membership_path": str(initial_path),
        "initial_source_path_count": len(source_paths),
        "initial_source_paths": ";".join(source_paths),
        "initial_partition_hash": _partition_hash(initial),
        "terminal_partition_hash": _partition_hash(final),
        "n_nodes": int(graph.n_nodes),
        "n_edges": int(graph.n_edges),
        "raw_n_clusters": int(raw.n_clusters),
        "post_nano_n_clusters": int(post_nano.n_clusters),
        "final_n_clusters": int(post_doc.n_clusters),
        "raw_quality_native": float(raw.quality),
        "raw_quality_weighted": float(raw_quality_weighted),
        "final_quality_weighted": float(final_quality),
        "raw_seconds": raw_seconds,
        "post_nano_seconds": post_nano_seconds,
        "post_doc_seconds": post_doc_seconds,
        "attempt_seconds": raw_seconds + post_nano_seconds + post_doc_seconds,
        "target_handle_count_scored": len(score_rows),
        **best_summary,
        "replay_execution_status": REPLAY_EXECUTION_STATUS,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "quality_cost_status": QUALITY_COST_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return attempt_row, score_rows


def _write_report(
    *,
    output_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    attempts: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Endpoint Replay Pilot Smoke",
        "",
        f"- readiness_dir: `{_rel(Path(config['readiness_dir']))}`",
        f"- status: `{summary['status']}`",
        f"- selected_attempt_count: {summary['selected_attempt_count']}",
        f"- executed_attempt_count: {summary['executed_attempt_count']}",
        f"- target_score_row_count: {summary['target_score_row_count']}",
        f"- branch_count: {summary['branch_count']}",
        f"- role_distinction_status: `{summary['role_distinction_status']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Best Family-Handle Scores",
    ]
    if attempts.empty:
        lines.append("- no attempts executed")
    else:
        for row in attempts.sort_values(["panel_case_rank", "role_side"]).itertuples(
            index=False
        ):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['attempt_id']}: final_clusters={data['final_n_clusters']}, "
                f"initial_exact={data['terminal_equals_initial_partition']}, "
                f"best_handle={data['best_target_handle_id']}, "
                f"best_weight_f1={data['best_cluster_weight_f1']}, "
                f"best_unit_f1={data['best_cluster_unit_f1']}, "
                f"seconds={data['attempt_seconds']:.3f}"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            (
                "This pilot confirms whether the endpoint-replay execution path is "
                "operational on the mirrored NanoClustering raw graph and produces "
                "terminal-to-family diagnostics. It is not a wall/pathway result "
                "because no route intervention has been executed."
            ),
            "",
        ]
    )
    (output_dir / PILOT_REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    readiness_dir = Path(args.readiness_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attempt_plan = _read_csv(readiness_dir / ATTEMPT_PLAN_ROWS_CSV)
    target_rows = _read_csv(readiness_dir / ENDPOINT_TARGET_ROWS_CSV)
    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    template_path = readiness_dir / RUNNER_CONFIG_TEMPLATE_JSON
    template = (
        json.loads(template_path.read_text(encoding="utf-8"))
        if template_path.exists()
        else {}
    )

    role_sides = _parse_csv_list(args.role_sides)
    method_seeds = _parse_csv_list(args.method_seeds, cast=int)
    case_ranks = _parse_csv_list(args.case_ranks, cast=int)
    selected_attempts = _select_attempts(
        attempt_plan,
        analysis_tier=args.analysis_tier,
        role_sides=role_sides,
        method_seeds=method_seeds,
        case_ranks=case_ranks,
        max_cases=int(args.max_cases),
    )

    graph_by_branch = {
        str(row["branch"]): row
        for _, row in graph_rows.iterrows()
        if str(row.get("runtime_graph_status", "")).startswith("ready_")
    }
    manifest_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    membership_cache: dict[str, np.ndarray] = {}
    mask_cache: dict[tuple[str, int], np.ndarray] = {}
    graph_cache: dict[str, tuple[RustLeidenGraph, np.ndarray, float]] = {}

    attempt_rows: list[dict[str, Any]] = []
    target_score_rows: list[dict[str, Any]] = []
    for attempt in selected_attempts.itertuples(index=False):
        attempt_series = pd.Series(attempt._asdict())
        branch = str(attempt_series["branch"])
        if branch not in graph_by_branch:
            raise ValueError(f"missing ready runtime graph for branch={branch}")
        if branch not in graph_cache:
            graph, _manifest, node_weights, load_seconds = _load_graph(
                graph_by_branch[branch],
                manifest_cache=manifest_cache,
            )
            graph_cache[branch] = (graph, node_weights, load_seconds)
        graph, node_weights, graph_load_seconds = graph_cache[branch]
        attempt_row, score_rows = _run_attempt(
            attempt=attempt_series,
            graph=graph,
            node_weights=node_weights,
            target_rows=target_rows,
            gamma=float(args.gamma),
            min_nano=int(args.min_nano),
            min_docs=float(args.min_docs),
            n_iterations=int(args.n_iterations),
            post_iterations=int(args.post_iterations),
            post_max_rounds=int(args.post_max_rounds),
            membership_cache=membership_cache,
            mask_cache=mask_cache,
        )
        attempt_row["graph_load_seconds_cached_branch"] = graph_load_seconds
        attempt_rows.append(attempt_row)
        target_score_rows.extend(score_rows)

    attempt_frame = pd.DataFrame(attempt_rows)
    target_score_frame = pd.DataFrame(target_score_rows)
    _write_csv(attempt_frame, output_dir / PILOT_ATTEMPT_ROWS_CSV)
    _write_csv(target_score_frame, output_dir / PILOT_TARGET_SCORE_ROWS_CSV)

    config = {
        "schema": "nanoclustering_endpoint_replay_pilot.v1",
        "readiness_dir": str(readiness_dir),
        "readiness_template": template,
        "output_dir": str(output_dir),
        "analysis_tier": args.analysis_tier,
        "role_sides": list(role_sides),
        "method_seeds": list(method_seeds),
        "case_ranks": list(case_ranks),
        "max_cases": int(args.max_cases),
        "gamma": float(args.gamma),
        "min_nano": int(args.min_nano),
        "min_docs": float(args.min_docs),
        "n_iterations": int(args.n_iterations),
        "post_iterations": int(args.post_iterations),
        "post_max_rounds": int(args.post_max_rounds),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / PILOT_CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if attempt_frame.empty:
        status = "no_selected_attempts"
    else:
        status = "executed_bounded_endpoint_replay_pilot"
    terminal_hash_unique_count = (
        int(attempt_frame["terminal_partition_hash"].nunique())
        if not attempt_frame.empty
        else 0
    )
    initial_hash_unique_count = (
        int(attempt_frame["initial_partition_hash"].nunique())
        if not attempt_frame.empty
        else 0
    )
    selected_role_count = (
        int(attempt_frame["role_side"].nunique()) if not attempt_frame.empty else 0
    )
    role_distinction_status = "not_evaluated_no_attempts"
    if not attempt_frame.empty and selected_role_count >= 2:
        role_distinction_status = (
            "blocked_terminal_partition_identical_across_roles"
            if terminal_hash_unique_count == 1
            else "terminal_partitions_distinct_across_roles"
        )
    elif not attempt_frame.empty:
        role_distinction_status = "single_role_smoke_no_role_distinction_test"
    summary = {
        "schema": "nanoclustering_endpoint_replay_pilot_summary.v1",
        "status": status,
        "readiness_dir": str(readiness_dir),
        "output_dir": str(output_dir),
        "selected_attempt_count": int(len(selected_attempts)),
        "executed_attempt_count": int(len(attempt_frame)),
        "target_score_row_count": int(len(target_score_frame)),
        "branch_count": (
            int(attempt_frame["branch"].nunique()) if not attempt_frame.empty else 0
        ),
        "initial_partition_hash_unique_count": initial_hash_unique_count,
        "terminal_partition_hash_unique_count": terminal_hash_unique_count,
        "role_distinction_status": role_distinction_status,
        "terminal_equals_initial_count": (
            int(attempt_frame["terminal_equals_initial_partition"].sum())
            if not attempt_frame.empty
            else 0
        ),
        "mean_best_cluster_weight_f1": (
            float(attempt_frame["best_cluster_weight_f1"].mean())
            if not attempt_frame.empty
            else None
        ),
        "max_best_cluster_weight_f1": (
            float(attempt_frame["best_cluster_weight_f1"].max())
            if not attempt_frame.empty
            else None
        ),
        "total_attempt_seconds": (
            float(attempt_frame["attempt_seconds"].sum())
            if not attempt_frame.empty
            else 0.0
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / PILOT_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        config=config,
        summary=summary,
        attempts=attempt_frame,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-tier", default="strict_core_v0_primary")
    parser.add_argument("--role-sides", default="candidate,control")
    parser.add_argument("--method-seeds", default="0")
    parser.add_argument("--case-ranks", default="")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--min-nano", type=int, default=3)
    parser.add_argument("--min-docs", type=float, default=3000.0)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--post-iterations", type=int, default=2)
    parser.add_argument("--post-max-rounds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
