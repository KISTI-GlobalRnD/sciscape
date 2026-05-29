"""Build paper-validation tables for hierarchy postprocess experiments.

This script intentionally reuses existing adaptive-refinement artifacts.  It
does not rerun Leiden.  The outputs are evidence tables for the claim that the
quality-preserving oversize postprocess improves the contraction input used by
hierarchical CPM Leiden.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
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


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_RESULTS_DIR = Path("research/consensus/results/adaptive_refinement")
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "hierarchy_postprocess_validation"

POLICY_ORDER = {
    "raw": 0,
    "small_only": 1,
    "oversize_split_only": 2,
    "two_stage_quality_first": 3,
    "two_stage_hard_cap": 4,
    "two_stage_hard_cap_aggressive": 5,
}

@dataclass(frozen=True)
class SampleConfig:
    sample: str
    sample_dir: Path
    prepare_summary_path: Path | None
    base_membership_path: Path | None
    node_weights_path: Path | None
    target_min_doc_weight: float | None
    target_max_doc_weight: float | None
    raw_n_clusters: int | None
    post_n_clusters: int | None
    n_nodes: int | None

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved

def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result

def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _policy_from_run(run: str) -> str:
    if run == "split_repair_only":
        return "oversize_split_only"
    if run == "quality_first_trim":
        return "two_stage_quality_first"
    if run.startswith("hard_cap_trim_neg"):
        return "two_stage_hard_cap_aggressive"
    if run.startswith("hard_cap_trim"):
        return "two_stage_hard_cap"
    return run

def _load_membership(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["node_idx", "cluster"])
    node_idx = table.column("node_idx").combine_chunks().to_numpy(zero_copy_only=False)
    cluster = table.column("cluster").combine_chunks().to_numpy(zero_copy_only=False)
    if node_idx.size and not np.all(node_idx[:-1] <= node_idx[1:]):
        cluster = cluster[np.argsort(node_idx, kind="stable")]
    return np.asarray(cluster, dtype=np.int64)

def _load_node_weights(path: Path | None, n_nodes: int) -> np.ndarray:
    if path is not None and path.exists():
        weights = np.fromfile(path, dtype=np.float64)
        if weights.shape[0] == n_nodes:
            return weights
    return np.ones(n_nodes, dtype=np.float64)

def _cluster_weights(membership: np.ndarray, node_weights: np.ndarray) -> np.ndarray:
    if membership.size == 0:
        return np.asarray([], dtype=np.float64)
    weights = np.bincount(
        membership,
        weights=np.asarray(node_weights, dtype=np.float64),
        minlength=int(membership.max()) + 1,
    )
    counts = np.bincount(membership, minlength=weights.shape[0])
    return np.asarray(weights[counts > 0], dtype=np.float64)

def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / total - (n + 1)) / n)

def _normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[values > 0]
    if values.size <= 1:
        return 0.0
    p = values / values.sum()
    entropy = -float(np.sum(p * np.log(p)))
    return float(entropy / math.log(values.size))

def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))

def _weight_metrics(
    membership_path: Path | None,
    node_weights_path: Path | None,
    *,
    target_max_doc_weight: float | None,
) -> dict[str, Any]:
    if membership_path is None or not membership_path.exists():
        return {}
    membership = _load_membership(membership_path)
    node_weights = _load_node_weights(node_weights_path, int(membership.shape[0]))
    weights = _cluster_weights(membership, node_weights)
    total = float(weights.sum()) if weights.size else 0.0
    sorted_desc = np.sort(weights)[::-1]
    target = float(target_max_doc_weight or 0.0)
    return {
        "n_nodes": int(membership.shape[0]),
        "n_clusters": int(weights.size),
        "total_doc_weight": total,
        "max_doc_weight": float(sorted_desc[0]) if sorted_desc.size else 0.0,
        "p50_doc_weight": _percentile(weights, 50),
        "p90_doc_weight": _percentile(weights, 90),
        "p95_doc_weight": _percentile(weights, 95),
        "p99_doc_weight": _percentile(weights, 99),
        "gini_doc_weight": _gini(weights),
        "entropy_doc_weight": _normalized_entropy(weights),
        "top1_doc_weight_share": float(sorted_desc[:1].sum() / total) if total else 0.0,
        "top5_doc_weight_share": float(sorted_desc[:5].sum() / total) if total else 0.0,
        "top10_doc_weights": [float(value) for value in sorted_desc[:10]],
        "n_above_target_max": int((weights > target).sum()) if target > 0.0 else 0,
        "target_max_satisfied": bool(target <= 0.0 or not np.any(weights > target)),
    }

def _membership_path_from_run_summary(
    run_summary: dict[str, Any] | None,
    *,
    artifact: str | None,
) -> Path | None:
    if not run_summary:
        return None
    paths = run_summary.get("paths", {})
    if artifact == "diagnostic_membership":
        return _repo_path(paths.get("diagnostic_membership"))
    if artifact == "applied_membership":
        return _repo_path(paths.get("applied_membership"))
    return _repo_path(paths.get("applied_membership") or paths.get("diagnostic_membership"))

def _sample_configs(results_dir: Path, cross_summary: dict[str, Any] | None) -> dict[str, SampleConfig]:
    configs: dict[str, SampleConfig] = {}
    policy_matrix_path = results_dir / "field12_postprocess_policy_matrix" / "policy_matrix_summary.json"
    field12_prepare = results_dir / "field12_gcc_split_repair_apply_pilot_top50" / "prepare_summary.json"
    if policy_matrix_path.exists():
        matrix = _read_json(policy_matrix_path)
        prepare = _read_json(field12_prepare) if field12_prepare.exists() else {}
        sample = str(matrix.get("sample") or prepare.get("sample") or "field12_gcc_emb_full_knn30")
        base_membership = _repo_path(matrix.get("membership"))
        graph_dir = _repo_path(matrix.get("graph_dir"))
        configs[sample] = SampleConfig(
            sample=sample,
            sample_dir=policy_matrix_path.parent,
            prepare_summary_path=field12_prepare if field12_prepare.exists() else None,
            base_membership_path=base_membership,
            node_weights_path=(graph_dir / "node_weights.f64.bin") if graph_dir else None,
            target_min_doc_weight=_safe_float(matrix.get("target_min_doc_weight") or prepare.get("min_size")),
            target_max_doc_weight=_safe_float(matrix.get("target_max_doc_weight")),
            raw_n_clusters=_safe_int(prepare.get("raw_n_clusters")),
            post_n_clusters=_safe_int(prepare.get("post_n_clusters")),
            n_nodes=_safe_int(prepare.get("n_nodes")),
        )

    if cross_summary:
        samples = sorted({str(row["sample"]) for row in cross_summary.get("runs", [])})
        cross_root = results_dir / "postprocess_policy_matrix_cross_sample"
        for sample in samples:
            if sample in configs:
                continue
            sample_dir = cross_root / sample
            prepare_path = sample_dir / "prepare_summary.json"
            prepare = _read_json(prepare_path) if prepare_path.exists() else {}
            paths = prepare.get("paths", {})
            graph_dir = _repo_path(paths.get("graph_dir")) or sample_dir / "graph"
            configs[sample] = SampleConfig(
                sample=sample,
                sample_dir=sample_dir,
                prepare_summary_path=prepare_path if prepare_path.exists() else None,
                base_membership_path=_repo_path(paths.get("membership")) or sample_dir / "membership.parquet",
                node_weights_path=graph_dir / "node_weights.f64.bin",
                target_min_doc_weight=_safe_float(prepare.get("min_size")),
                target_max_doc_weight=_safe_float(prepare.get("target_max_doc_weight")),
                raw_n_clusters=_safe_int(prepare.get("raw_n_clusters")),
                post_n_clusters=_safe_int(prepare.get("post_n_clusters")),
                n_nodes=_safe_int(prepare.get("n_nodes")),
            )
    return configs

def _iter_run_rows(results_dir: Path) -> tuple[list[dict[str, Any]], dict[str, SampleConfig]]:
    cross_path = results_dir / "postprocess_policy_matrix_cross_sample" / "cross_sample_summary.json"
    cross_summary = _read_json(cross_path) if cross_path.exists() else None
    configs = _sample_configs(results_dir, cross_summary)

    raw_runs: list[dict[str, Any]] = []
    if cross_summary:
        for row in cross_summary.get("runs", []):
            current = dict(row)
            current["source_summary"] = str(cross_path)
            raw_runs.append(current)
    else:
        policy_matrix_path = results_dir / "field12_postprocess_policy_matrix" / "policy_matrix_summary.json"
        if policy_matrix_path.exists():
            matrix = _read_json(policy_matrix_path)
            for row in matrix.get("runs", []):
                current = dict(row)
                current["sample"] = str(matrix.get("sample", "field12_gcc_emb_full_knn30"))
                current["source_summary"] = str(policy_matrix_path)
                raw_runs.append(current)
    return raw_runs, configs

def _build_eval_tables(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_runs, configs = _iter_run_rows(results_dir)
    inventory_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    contraction_rows: list[dict[str, Any]] = []
    membership_metric_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def metrics_for(sample: str, path: Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        key = (sample, str(path))
        if key not in membership_metric_cache:
            cfg = configs[sample]
            membership_metric_cache[key] = _weight_metrics(
                path,
                cfg.node_weights_path,
                target_max_doc_weight=cfg.target_max_doc_weight,
            )
        return membership_metric_cache[key]

    for sample, cfg in sorted(configs.items()):
        inventory_rows.append(
            {
                "sample": sample,
                "policy": "raw",
                "run": "raw",
                "artifact_kind": "prepare_summary",
                "source_summary": _rel(cfg.prepare_summary_path),
                "output_dir": None,
                "membership_path": None,
                "diagnostic_membership_path": None,
                "summary_exists": bool(cfg.prepare_summary_path and cfg.prepare_summary_path.exists()),
                "membership_exists": False,
                "status": "summary_only",
                "target_max_doc_weight": cfg.target_max_doc_weight,
            }
        )
        eval_rows.append(
            {
                "sample": sample,
                "level": 0,
                "policy": "raw",
                "run": "raw",
                "status": "summary_only",
                "accepted_for_contraction": False,
                "fallback_used": False,
                "target_min_doc_weight": cfg.target_min_doc_weight,
                "target_max_doc_weight": cfg.target_max_doc_weight,
                "delta_q_reported": None,
                "effective_delta_q": None,
                "initial_n_clusters": cfg.raw_n_clusters,
                "reported_final_n_clusters": cfg.raw_n_clusters,
                "effective_n_clusters": cfg.raw_n_clusters,
            }
        )

        base_metrics = metrics_for(sample, cfg.base_membership_path)
        inventory_rows.append(
            {
                "sample": sample,
                "policy": "small_only",
                "run": "small_only",
                "artifact_kind": "membership",
                "source_summary": _rel(cfg.prepare_summary_path),
                "output_dir": _rel(cfg.sample_dir),
                "membership_path": _rel(cfg.base_membership_path),
                "diagnostic_membership_path": None,
                "summary_exists": bool(cfg.prepare_summary_path and cfg.prepare_summary_path.exists()),
                "membership_exists": bool(cfg.base_membership_path and cfg.base_membership_path.exists()),
                "status": "committed",
                "target_max_doc_weight": cfg.target_max_doc_weight,
            }
        )
        eval_rows.append(
            _eval_row_from_metrics(
                sample=sample,
                policy="small_only",
                run="small_only",
                status="committed",
                accepted=True,
                fallback_used=False,
                target_min=cfg.target_min_doc_weight,
                target_max=cfg.target_max_doc_weight,
                initial_metrics=base_metrics,
                reported_metrics=base_metrics,
                effective_metrics=base_metrics,
                delta_q_reported=0.0,
                effective_delta_q=0.0,
                output_dir=cfg.sample_dir,
                membership_path=cfg.base_membership_path,
                diagnostic_membership_path=None,
                stop_reason="baseline_after_small_cluster_repair",
            )
        )
        contraction_rows.append(
            _contraction_row(
                sample=sample,
                policy="small_only",
                run="small_only",
                accepted=True,
                fallback_used=False,
                target_max=cfg.target_max_doc_weight,
                effective_metrics=base_metrics,
                reported_metrics=base_metrics,
                baseline_metrics=base_metrics,
            )
        )

    for raw in raw_runs:
        sample = str(raw["sample"])
        if sample not in configs:
            continue
        cfg = configs[sample]
        run = str(raw.get("run", ""))
        policy = _policy_from_run(run)
        output_dir = _repo_path(raw.get("output_dir"))
        summary_path = output_dir / "iterative_split_repair_apply_summary.json" if output_dir else None
        run_summary = _read_json(summary_path) if summary_path and summary_path.exists() else None
        artifact = raw.get("membership_artifact")
        proposed_membership_path = _membership_path_from_run_summary(
            run_summary,
            artifact=str(artifact) if artifact else None,
        )
        diagnostic_membership_path = _membership_path_from_run_summary(
            run_summary,
            artifact="diagnostic_membership",
        )
        status = str(raw.get("status") or (run_summary or {}).get("status") or "")
        accepted = status == "committed"
        effective_membership_path = proposed_membership_path if accepted else cfg.base_membership_path
        fallback_used = not accepted
        initial_metrics = metrics_for(sample, cfg.base_membership_path)
        reported_metrics = metrics_for(sample, proposed_membership_path)
        if not reported_metrics:
            reported_metrics = _metrics_from_run_fields(raw, prefix="final")
        effective_metrics = metrics_for(sample, effective_membership_path)
        inventory_rows.append(
            {
                "sample": sample,
                "policy": policy,
                "run": run,
                "artifact_kind": "run_summary",
                "source_summary": _rel(_repo_path(raw.get("source_summary"))),
                "output_dir": _rel(output_dir),
                "membership_path": _rel(effective_membership_path),
                "diagnostic_membership_path": _rel(diagnostic_membership_path),
                "summary_exists": bool(summary_path and summary_path.exists()),
                "membership_exists": bool(
                    effective_membership_path and effective_membership_path.exists()
                ),
                "status": status,
                "target_max_doc_weight": cfg.target_max_doc_weight,
            }
        )
        eval_rows.append(
            _eval_row_from_metrics(
                sample=sample,
                policy=policy,
                run=run,
                status=status,
                accepted=accepted,
                fallback_used=fallback_used,
                target_min=cfg.target_min_doc_weight,
                target_max=cfg.target_max_doc_weight,
                initial_metrics=initial_metrics or _metrics_from_run_fields(raw, prefix="initial"),
                reported_metrics=reported_metrics,
                effective_metrics=effective_metrics,
                delta_q_reported=_safe_float(raw.get("final_exact_delta_q")),
                effective_delta_q=_safe_float(raw.get("final_exact_delta_q")) if accepted else 0.0,
                output_dir=output_dir,
                membership_path=effective_membership_path,
                diagnostic_membership_path=diagnostic_membership_path,
                stop_reason=str(raw.get("stop_reason") or (run_summary or {}).get("stop_reason") or ""),
                run_summary=run_summary,
            )
        )
        contraction_rows.append(
            _contraction_row(
                sample=sample,
                policy=policy,
                run=run,
                accepted=accepted,
                fallback_used=fallback_used,
                target_max=cfg.target_max_doc_weight,
                effective_metrics=effective_metrics,
                reported_metrics=reported_metrics,
                baseline_metrics=initial_metrics,
            )
        )

    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["sample", "policy"],
        key=lambda s: s.map(POLICY_ORDER).fillna(99) if s.name == "policy" else s,
    )
    eval_df = pd.DataFrame(eval_rows)
    eval_df["policy_order"] = eval_df["policy"].map(POLICY_ORDER).fillna(99).astype(int)
    eval_df = eval_df.sort_values(["sample", "policy_order", "run"]).drop(columns=["policy_order"])
    contraction = pd.DataFrame(contraction_rows)
    contraction["policy_order"] = contraction["policy"].map(POLICY_ORDER).fillna(99).astype(int)
    contraction = contraction.sort_values(["sample", "policy_order", "run"]).drop(columns=["policy_order"])
    return inventory, eval_df, contraction

def _metrics_from_run_fields(row: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    return {
        "n_clusters": _safe_int(row.get(f"{prefix}_n_clusters")),
        "max_doc_weight": _safe_float(row.get(f"{prefix}_max_doc_weight")),
        "n_above_target_max": _safe_int(row.get(f"{prefix}_n_above_max_doc_weight")),
        "target_max_satisfied": (_safe_int(row.get(f"{prefix}_n_above_max_doc_weight")) == 0),
    }

def _eval_row_from_metrics(
    *,
    sample: str,
    policy: str,
    run: str,
    status: str,
    accepted: bool,
    fallback_used: bool,
    target_min: float | None,
    target_max: float | None,
    initial_metrics: dict[str, Any],
    reported_metrics: dict[str, Any],
    effective_metrics: dict[str, Any],
    delta_q_reported: float | None,
    effective_delta_q: float | None,
    output_dir: Path | None,
    membership_path: Path | None,
    diagnostic_membership_path: Path | None,
    stop_reason: str,
    run_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    initial_max = _safe_float(initial_metrics.get("max_doc_weight"))
    reported_max = _safe_float(reported_metrics.get("max_doc_weight"))
    effective_max = _safe_float(effective_metrics.get("max_doc_weight"))
    target = float(target_max or 0.0)
    return {
        "sample": sample,
        "level": 0,
        "policy": policy,
        "run": run,
        "status": status,
        "stop_reason": stop_reason,
        "accepted_for_contraction": bool(accepted),
        "fallback_used": bool(fallback_used),
        "target_min_doc_weight": target_min,
        "target_max_doc_weight": target_max,
        "delta_q_reported": delta_q_reported,
        "effective_delta_q": effective_delta_q,
        "output_dir": _rel(output_dir),
        "membership_path": _rel(membership_path),
        "diagnostic_membership_path": _rel(diagnostic_membership_path),
        "initial_n_clusters": initial_metrics.get("n_clusters"),
        "reported_final_n_clusters": reported_metrics.get("n_clusters"),
        "effective_n_clusters": effective_metrics.get("n_clusters"),
        "initial_max_doc_weight": initial_max,
        "reported_final_max_doc_weight": reported_max,
        "effective_max_doc_weight": effective_max,
        "initial_n_above_max_doc_weight": initial_metrics.get("n_above_target_max"),
        "reported_final_n_above_max_doc_weight": reported_metrics.get("n_above_target_max"),
        "effective_n_above_max_doc_weight": effective_metrics.get("n_above_target_max"),
        "reported_target_max_satisfied": bool(reported_metrics.get("target_max_satisfied", False)),
        "effective_target_max_satisfied": bool(effective_metrics.get("target_max_satisfied", False)),
        "reported_max_ratio": (reported_max / target) if target > 0.0 and reported_max is not None else None,
        "effective_max_ratio": (effective_max / target) if target > 0.0 and effective_max is not None else None,
        "reported_max_reduction": (
            initial_max - reported_max
            if initial_max is not None and reported_max is not None
            else None
        ),
        "effective_max_reduction": (
            initial_max - effective_max
            if initial_max is not None and effective_max is not None
            else None
        ),
        "reported_gini_doc_weight": reported_metrics.get("gini_doc_weight"),
        "effective_gini_doc_weight": effective_metrics.get("gini_doc_weight"),
        "reported_entropy_doc_weight": reported_metrics.get("entropy_doc_weight"),
        "effective_entropy_doc_weight": effective_metrics.get("entropy_doc_weight"),
        "reported_p95_doc_weight": reported_metrics.get("p95_doc_weight"),
        "effective_p95_doc_weight": effective_metrics.get("p95_doc_weight"),
        "reported_p99_doc_weight": reported_metrics.get("p99_doc_weight"),
        "effective_p99_doc_weight": effective_metrics.get("p99_doc_weight"),
        "reported_top1_doc_weight_share": reported_metrics.get("top1_doc_weight_share"),
        "effective_top1_doc_weight_share": effective_metrics.get("top1_doc_weight_share"),
        "split_repair_exact_delta_q": _nested_float(run_summary, "split_repair_exact_delta_q"),
        "trim_exact_delta_q": _nested_float(run_summary, "trim_exact_delta_q"),
        "trim_moves": _nested_int(run_summary, "trim", "n_moves"),
        "trim_moves_proposed": _nested_int(run_summary, "trim", "n_moves_proposed"),
        "quality_floor_limited": _nested_bool(run_summary, "trim", "quality_floor_limited"),
    }

def _contraction_row(
    *,
    sample: str,
    policy: str,
    run: str,
    accepted: bool,
    fallback_used: bool,
    target_max: float | None,
    effective_metrics: dict[str, Any],
    reported_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    baseline_max = _safe_float(baseline_metrics.get("max_doc_weight"))
    effective_max = _safe_float(effective_metrics.get("max_doc_weight"))
    reported_max = _safe_float(reported_metrics.get("max_doc_weight"))
    target = float(target_max or 0.0)
    return {
        "sample": sample,
        "level": 0,
        "transition": "level0_to_level1_precondition",
        "scope": "contraction_input_only_no_next_leiden_rerun",
        "policy": policy,
        "run": run,
        "accepted_for_contraction": bool(accepted),
        "fallback_used": bool(fallback_used),
        "target_max_doc_weight": target_max,
        "effective_supernode_count": effective_metrics.get("n_clusters"),
        "effective_max_supernode_doc_weight": effective_max,
        "effective_p95_supernode_doc_weight": effective_metrics.get("p95_doc_weight"),
        "effective_p99_supernode_doc_weight": effective_metrics.get("p99_doc_weight"),
        "effective_supernode_gini": effective_metrics.get("gini_doc_weight"),
        "effective_top1_doc_weight_share": effective_metrics.get("top1_doc_weight_share"),
        "effective_n_supernodes_above_current_target": effective_metrics.get("n_above_target_max"),
        "reported_max_supernode_doc_weight": reported_max,
        "reported_n_supernodes_above_current_target": reported_metrics.get("n_above_target_max"),
        "max_supernode_reduction_vs_small": (
            baseline_max - effective_max
            if baseline_max is not None and effective_max is not None
            else None
        ),
        "reported_max_supernode_reduction_vs_small": (
            baseline_max - reported_max
            if baseline_max is not None and reported_max is not None
            else None
        ),
        "effective_max_supernode_ratio": (
            effective_max / target if target > 0.0 and effective_max is not None else None
        ),
    }

def _nested_float(payload: dict[str, Any] | None, *keys: str) -> float | None:
    value = _nested_value(payload, *keys)
    return _safe_float(value)

def _nested_int(payload: dict[str, Any] | None, *keys: str) -> int | None:
    value = _nested_value(payload, *keys)
    return _safe_int(value)

def _nested_bool(payload: dict[str, Any] | None, *keys: str) -> bool | None:
    value = _nested_value(payload, *keys)
    if value is None:
        return None
    return bool(value)

def _nested_value(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

def _build_policy_comparison(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = eval_df[eval_df["policy"].isin(POLICY_ORDER) & (eval_df["policy"] != "raw")]
    for policy, group in subset.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "n_runs": int(len(group)),
                "accepted_rate": float(group["accepted_for_contraction"].mean()),
                "fallback_rate": float(group["fallback_used"].mean()),
                "reported_target_satisfied_rate": float(group["reported_target_max_satisfied"].mean()),
                "effective_target_satisfied_rate": float(group["effective_target_max_satisfied"].mean()),
                "mean_delta_q_reported": float(group["delta_q_reported"].dropna().mean())
                if group["delta_q_reported"].notna().any()
                else None,
                "median_delta_q_reported": float(group["delta_q_reported"].dropna().median())
                if group["delta_q_reported"].notna().any()
                else None,
                "mean_reported_max_ratio": float(group["reported_max_ratio"].dropna().mean())
                if group["reported_max_ratio"].notna().any()
                else None,
                "mean_effective_max_ratio": float(group["effective_max_ratio"].dropna().mean())
                if group["effective_max_ratio"].notna().any()
                else None,
                "mean_effective_max_reduction": float(group["effective_max_reduction"].dropna().mean())
                if group["effective_max_reduction"].notna().any()
                else None,
            }
        )
    out = pd.DataFrame(rows)
    out["policy_order"] = out["policy"].map(POLICY_ORDER).fillna(99).astype(int)
    return out.sort_values("policy_order").drop(columns=["policy_order"])

def _build_failure_taxonomy(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for row in eval_df.to_dict("records"):
        if row["policy"] in {"raw", "small_only"}:
            continue
        if bool(row.get("reported_target_max_satisfied")):
            tags = ["resolved"]
        else:
            tags = _failure_tags(row)
        for tag in tags:
            rows.append(
                {
                    "sample": row["sample"],
                    "policy": row["policy"],
                    "run": row["run"],
                    "status": row["status"],
                    "failure_tag": tag,
                    "is_primary": tag == tags[0],
                    "final_n_above_max_doc_weight": row.get("reported_final_n_above_max_doc_weight"),
                    "final_max_doc_weight": row.get("reported_final_max_doc_weight"),
                    "target_max_doc_weight": row.get("target_max_doc_weight"),
                    "delta_q_reported": row.get("delta_q_reported"),
                    "fallback_used": row.get("fallback_used"),
                    "next_step": _failure_next_step(tag),
                }
            )
    taxonomy = pd.DataFrame(rows)
    if taxonomy.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            taxonomy.groupby("failure_tag", as_index=False)
            .agg(
                n_cases=("failure_tag", "size"),
                primary_cases=("is_primary", "sum"),
                fallback_cases=("fallback_used", "sum"),
                mean_delta_q=("delta_q_reported", "mean"),
                max_final_oversize=("final_max_doc_weight", "max"),
                next_step=("next_step", "first"),
            )
            .sort_values(["primary_cases", "n_cases", "failure_tag"], ascending=[False, False, True])
        )
    return taxonomy, summary

def _failure_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if bool(row.get("fallback_used")):
        tags.append("fallback_required")
    if bool(row.get("quality_floor_limited")):
        tags.append("quality_floor_limited")
    stop_reason = str(row.get("stop_reason") or "")
    if stop_reason in {"no_selected_candidates", "max_iterations_reached"}:
        tags.append("insufficient_split_repair_candidates")
    trim_moves = _safe_int(row.get("trim_moves")) or 0
    trim_moves_proposed = _safe_int(row.get("trim_moves_proposed")) or 0
    if trim_moves_proposed > 0 and trim_moves == trim_moves_proposed:
        tags.append("boundary_too_dense_or_receiver_cap")
    if _safe_float(row.get("reported_max_ratio")) and float(row["reported_max_ratio"]) > 1.05:
        tags.append("semantic_core_cluster")
    if not tags:
        tags.append("residual_oversize_unclassified")
    return list(dict.fromkeys(tags))

def _failure_next_step(tag: str) -> str:
    mapping = {
        "resolved": "Use as positive case in method table.",
        "fallback_required": "Keep hard-cap as diagnostic unless target feasibility improves.",
        "quality_floor_limited": "Probe adaptive quality budgets or local resolution before trim.",
        "insufficient_split_repair_candidates": "Expand local gamma/pair-seeded probes for dense residual clusters.",
        "boundary_too_dense_or_receiver_cap": "Diagnose receiver-cap constraints and near-neutral boundary alternatives.",
        "semantic_core_cluster": "Treat as stable broad topic or split with semantic axes, not pure CPM trim.",
        "residual_oversize_unclassified": "Inspect cluster-level diagnostics manually.",
    }
    return mapping.get(tag, "Inspect manually.")

def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    df.to_parquet(path.with_suffix(".parquet"), index=False)

def _write_markdown_table(df: pd.DataFrame, path: Path, *, max_rows: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    view = df.head(max_rows) if max_rows else df
    path.write_text(_markdown_table(view) + "\n", encoding="utf-8")

def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.to_dict("records"):
        values = [_markdown_cell(row.get(column)) for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")

def _plot_pipeline(out_dir: Path) -> Path:
    labels = [
        "Raw CPM Leiden",
        "Small repair",
        "Oversize split-repair",
        "Boundary trim",
        "Contraction input",
    ]
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off")
    x_positions = np.linspace(0.08, 0.92, len(labels))
    for idx, (x, label) in enumerate(zip(x_positions, labels, strict=True)):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.04",
                "facecolor": "#f5f5f5",
                "edgecolor": "#333333",
            },
            fontsize=10,
        )
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.085, 0.55),
                xytext=(x + 0.085, 0.55),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"},
            )
    ax.text(
        0.5,
        0.16,
        "quality_first accepts by exact CPM floor; hard_cap also requires max doc-weight target",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    out_path = out_dir / "figure1_two_stage_pipeline.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _plot_tradeoff(eval_df: pd.DataFrame, out_dir: Path) -> Path:
    subset = eval_df[
        eval_df["policy"].isin(
            ["small_only", "oversize_split_only", "two_stage_quality_first", "two_stage_hard_cap"]
        )
    ].copy()
    colors = {
        "small_only": "#666666",
        "oversize_split_only": "#4c78a8",
        "two_stage_quality_first": "#54a24b",
        "two_stage_hard_cap": "#e45756",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for policy, group in subset.groupby("policy", sort=False):
        ax.scatter(
            group["delta_q_reported"].fillna(0.0),
            group["reported_max_ratio"],
            label=policy,
            s=70,
            alpha=0.9,
            color=colors.get(policy),
            edgecolor="white",
            linewidth=0.8,
        )
        for _, item in group.iterrows():
            ax.annotate(
                str(item["sample"]).replace("_gcc_emb_full_knn30", "").replace("_combo_dc_bc_cc_sum", ""),
                (item["delta_q_reported"] if pd.notna(item["delta_q_reported"]) else 0.0, item["reported_max_ratio"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
            )
    ax.axhline(1.0, color="#222222", linewidth=1, linestyle="--")
    ax.set_xlabel("reported exact delta Q vs small-only baseline")
    ax.set_ylabel("reported max doc weight / target max")
    ax.set_title("Size imbalance vs quality tradeoff")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "figure2_size_quality_tradeoff.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def _plot_contraction(contraction: pd.DataFrame, out_dir: Path) -> Path:
    subset = contraction[
        contraction["policy"].isin(
            ["oversize_split_only", "two_stage_quality_first", "two_stage_hard_cap"]
        )
    ].copy()
    subset["reduction_pct"] = (
        subset["max_supernode_reduction_vs_small"]
        / subset.groupby("sample")["effective_max_supernode_doc_weight"].transform(
            lambda values: values.iloc[0] if len(values) else np.nan
        )
    )
    # The transform above is only a display helper; recompute against per-sample
    # small-only max from the full table for correct denominators.
    baselines = contraction[contraction["policy"] == "small_only"].set_index("sample")[
        "effective_max_supernode_doc_weight"
    ]
    subset["reduction_pct"] = [
        100.0 * row["max_supernode_reduction_vs_small"] / baselines[row["sample"]]
        if row["sample"] in baselines and baselines[row["sample"]] > 0
        else np.nan
        for _, row in subset.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [str(x) for x in subset["sample"] + "\n" + subset["policy"]]
    ax.bar(np.arange(len(subset)), subset["reduction_pct"].fillna(0.0), color="#4c78a8")
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.set_xticks(np.arange(len(subset)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("effective max supernode reduction vs small-only (%)")
    ax.set_title("Contraction input imbalance reduction")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out_path = out_dir / "figure3_contraction_precondition.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def _write_report(
    out_dir: Path,
    *,
    inventory: pd.DataFrame,
    eval_df: pd.DataFrame,
    policy_comparison: pd.DataFrame,
    taxonomy_summary: pd.DataFrame,
    figures: list[Path],
) -> Path:
    qf = eval_df[eval_df["policy"] == "two_stage_quality_first"]
    hc = eval_df[eval_df["policy"] == "two_stage_hard_cap"]
    lines = [
        "# Hierarchy Postprocess Validation",
        "",
        "This report is generated from existing adaptive-refinement artifacts; no new Leiden rerun was performed.",
        "",
        "## Artifact Coverage",
        "",
        f"- Inventory rows: {len(inventory)}",
        f"- Evaluation rows: {len(eval_df)}",
        f"- Samples: {', '.join(sorted(eval_df['sample'].dropna().unique()))}",
        "",
        "## Main Readout",
        "",
        f"- quality_first runs: {len(qf)}, accepted rate {qf['accepted_for_contraction'].mean():.2f}, "
        f"mean reported delta Q {qf['delta_q_reported'].mean():.3f}",
        f"- hard_cap default runs: {len(hc)}, accepted rate {hc['accepted_for_contraction'].mean():.2f}, "
        f"reported target satisfaction {hc['reported_target_max_satisfied'].mean():.2f}",
        "- Contraction evidence is a precondition metric: it compares the supernode weight distribution that would be passed to the next level, not a fresh next-level Leiden run.",
        "",
        "## Generated Tables",
        "",
        "- `available_runs.csv`",
        "- `hierarchy_postprocess_eval.csv` / `.parquet`",
        "- `contraction_effects.csv` / `.parquet`",
        "- `policy_comparison.csv` / `.parquet`",
        "- `failure_taxonomy.csv` / `.parquet`",
        "- `table1_policy_comparison.md`",
        "- `table2_failure_taxonomy.md`",
        "",
        "## Figures",
        "",
    ]
    for figure in figures:
        lines.append(f"- `{figure.name}`")
    if not policy_comparison.empty:
        lines.extend(["", "## Policy Comparison", "", _markdown_table(policy_comparison)])
    if not taxonomy_summary.empty:
        lines.extend(["", "## Failure Taxonomy", "", _markdown_table(taxonomy_summary)])
    report_path = out_dir / "README.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    results_dir = _repo_path(args.results_dir)
    assert results_dir is not None
    output_dir = _repo_path(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory, eval_df, contraction = _build_eval_tables(results_dir)
    policy_comparison = _build_policy_comparison(eval_df)
    taxonomy, taxonomy_summary = _build_failure_taxonomy(eval_df)

    inventory.to_csv(output_dir / "available_runs.csv", index=False)
    _write_table(eval_df, output_dir / "hierarchy_postprocess_eval")
    _write_table(contraction, output_dir / "contraction_effects")
    _write_table(policy_comparison, output_dir / "policy_comparison")
    _write_table(taxonomy, output_dir / "failure_taxonomy")
    _write_table(taxonomy_summary, output_dir / "failure_taxonomy_summary")
    _write_markdown_table(policy_comparison, output_dir / "table1_policy_comparison.md")
    _write_markdown_table(taxonomy_summary, output_dir / "table2_failure_taxonomy.md")

    figures: list[Path] = []
    if not args.skip_figures:
        figures = [
            _plot_pipeline(output_dir),
            _plot_tradeoff(eval_df, output_dir),
            _plot_contraction(contraction, output_dir),
        ]

    report_path = _write_report(
        output_dir,
        inventory=inventory,
        eval_df=eval_df,
        policy_comparison=policy_comparison,
        taxonomy_summary=taxonomy_summary,
        figures=figures,
    )
    print(f"Saved hierarchy postprocess validation outputs to {_rel(output_dir)}")
    print(f"Report: {_rel(report_path)}")

if __name__ == "__main__":
    main()
