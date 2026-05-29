"""Pilot integrated Dongdaemun refinement on a prepared source-level graph.

This runner compares three variants on the same cached Rust graph:

- standard Leiden
- integrated Dongdaemun refinement with baseline repair disabled
- integrated Dongdaemun refinement with baseline repair enabled

The pilot is diagnostics-first.  It records equality, objective, oversize, and
audit fields so Slice 4 can be checked for safety on real prepared source-level
graphs before broader field expansion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
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


import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sciscape.clustering.leiden_rust import (  # noqa: E402
    DEFAULT_DONGDAEMUN_GAMMA_MULTIPLIERS,
    RUST_DONGDAEMUN_REFINEMENT_AVAILABLE,
    build_leiden_graph,
)

DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_refinement_slice4_pilot"
)
SCHEMA_VERSION = 1

VARIANT_STANDARD = "standard"
VARIANT_REPAIR_OFF = "refine_repair_off"
VARIANT_REPAIR_ON = "refine_repair_on"

AUDIT_INT_FIELDS = [
    "selected_parent_count_total",
    "applied_parent_count_total",
    "rejected_candidate_count_total",
    "added_refined_clusters_total",
    "same_gamma_candidates_total",
    "high_gamma_candidates_total",
    "same_gamma_applied_total",
    "high_gamma_applied_total",
    "quotient_candidates_total",
    "quotient_positive_candidates_total",
    "quotient_selected_total",
    "baseline_repair_candidates_total",
    "baseline_repair_improved_candidates_total",
    "baseline_repair_selected_total",
    "baseline_repair_merge_count_total",
    "candidate_positive_quality_delta_total",
    "candidate_selected_positive_quality_delta_total",
    "candidate_rejected_by_quality_total",
    "same_gamma_positive_quality_delta_total",
    "high_gamma_positive_quality_delta_total",
    "same_gamma_selected_positive_quality_delta_total",
    "high_gamma_selected_positive_quality_delta_total",
    "same_gamma_rejected_by_quality_total",
    "high_gamma_rejected_by_quality_total",
    "candidate_valid_total",
    "candidate_invalid_total",
    "candidate_rejected_by_policy_total",
    "same_gamma_valid_total",
    "high_gamma_valid_total",
    "same_gamma_invalid_total",
    "high_gamma_invalid_total",
    "same_gamma_rejected_by_policy_total",
    "high_gamma_rejected_by_policy_total",
    "candidate_qpos_spos_total",
    "candidate_qpos_sneg_total",
    "candidate_qneg_spos_total",
    "candidate_qneg_sneg_total",
    "same_gamma_qpos_spos_total",
    "same_gamma_qpos_sneg_total",
    "same_gamma_qneg_spos_total",
    "same_gamma_qneg_sneg_total",
    "high_gamma_qpos_spos_total",
    "high_gamma_qpos_sneg_total",
    "high_gamma_qneg_spos_total",
    "high_gamma_qneg_sneg_total",
    "candidate_true_positive_total",
    "candidate_false_positive_total",
    "candidate_false_negative_total",
    "candidate_true_negative_total",
    "adaptive_local_shake_triggers_total",
    "adaptive_local_shake_candidates_total",
    "adaptive_local_shake_commits_total",
    "final_quality_guard_enabled",
    "final_quality_guard_triggered",
]
AUDIT_FLOAT_FIELDS = [
    "quotient_score_sum",
    "baseline_repair_delta_sum",
    "candidate_quality_delta_sum",
    "same_gamma_quality_delta_sum",
    "high_gamma_quality_delta_sum",
    "adaptive_local_shake_qf_gain_sum",
    "final_quality_guard_standard_quality",
    "final_quality_guard_pre_guard_quality",
    "final_quality_delta_vs_guard_standard",
    "max_parent_weight_seen",
]
BASELINE_REPAIR_AUDIT_FIELDS = [
    "baseline_repair_candidates_total",
    "baseline_repair_improved_candidates_total",
    "baseline_repair_selected_total",
    "baseline_repair_merge_count_total",
    "baseline_repair_delta_sum",
]

CSV_FIELDS = [
    "sample",
    "variant",
    "supported",
    "unsupported_reason",
    "elapsed_sec",
    "n_clusters",
    "quality",
    "quality_delta_vs_standard",
    "quality_delta_vs_repair_off",
    "quality_improved_vs_standard",
    "quality_improved_vs_repair_off",
    "max_doc_weight",
    "max_doc_weight_ratio",
    "n_above_max_doc_weight",
    "top10_doc_weights",
    "membership_equal_to_standard",
    "membership_diff_nodes_vs_standard",
    "membership_equal_repair_off_on",
    "audit_enabled",
    "n_iterations_used",
    "audit_iteration_count",
    *AUDIT_INT_FIELDS,
    *AUDIT_FLOAT_FIELDS,
]

@dataclass(frozen=True)
class Slice4Input:
    sample: str
    graph_dir: Path
    membership_path: Path | None
    node_weights_path: Path
    resolution: float
    target_max_doc_weight: float
    seed: int
    n_nodes: int | None = None
    summary_path: Path | None = None

@dataclass(frozen=True)
class Slice4RunConfig:
    n_iterations: int = 10
    randomness: float = 0.01
    soft_min_ratio: float = 1.0
    max_extra_parents_per_iteration: int = 16
    max_extra_children_per_parent: int = 64
    parent_selection_policy: str = "weight"
    max_singleton_weight_fraction: float = 0.05
    min_largest_child_fraction_improvement: float = 0.05
    gamma_multipliers: tuple[float, ...] = DEFAULT_DONGDAEMUN_GAMMA_MULTIPLIERS
    seed_perturbations: int = 0
    use_quotient_diagnostic: bool = True
    baseline_repair_policy: str = "replace"
    baseline_repair_replace_min_parent_ratio: float = 1.05
    baseline_repair_epsilon: float = 0.0
    candidate_quality_policy: str = "structural"
    min_candidate_delta_q: float = 0.0
    adaptive_plateau_quality_band: float = 0.0
    use_final_quality_guard: bool = False
    min_final_quality_delta: float = 0.0
    adaptive_probe_mode: str = "off"
    adaptive_probe_perturbations: int = 0
    adaptive_probe_targets: tuple[str, ...] = ()
    adaptive_probe_tolerance_parent_weight: float = 1e-6
    adaptive_probe_include_node_order_control: bool = False
    adaptive_probe_commit_min_gain_parent_weight: float = 0.0
    adaptive_probe_max_commits_total: int = 0
    adaptive_probe_max_commits_per_depth: int = 0
    adaptive_probe_commit_sources: tuple[str, ...] = ()
    adaptive_probe_commit_strategy: str = "online_first"
    adaptive_near_tie_probe_mode: str = "off"
    adaptive_near_tie_margin_parent_weight: float = 0.0
    adaptive_near_tie_randomness: float = 0.0
    adaptive_near_tie_max_decisions_per_parent: int = 0
    adaptive_local_shake_mode: str = "off"
    adaptive_local_shake_arms: tuple[str, ...] = ()
    adaptive_local_shake_max_arms_per_parent: int = 0
    adaptive_local_shake_max_candidates_per_parent: int = 0
    adaptive_local_shake_min_gain_parent_weight: float = 0.0
    adaptive_local_shake_shape_eps: float = 1e-12
    adaptive_local_shake_arm_priority: tuple[str, ...] = ()
    adaptive_local_shake_near_tie_min_count: int = 1
    adaptive_local_shake_resolution_down_multipliers: tuple[float, ...] = ()
    adaptive_local_shake_resolution_up_multipliers: tuple[float, ...] = ()
    adaptive_local_shake_resolution_up_min_parent_ratio: float = 1.0
    adaptive_local_shake_resolution_down_max_parent_ratio: float = 1.0
    adaptive_local_shake_large_child_fraction: float = 0.95
    adaptive_local_shake_singleton_fraction: float = 0.05
    adaptive_local_shake_seed_perturbations: int = 0
    adaptive_local_shake_seed_margin_count: int = 2
    adaptive_local_shake_near_tie_margin_parent_weight: float = 0.0
    adaptive_local_shake_near_tie_randomness: float = 0.0
    adaptive_local_shake_final_guard_mode: str = "none"

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

def _repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    return REPO_ROOT / parsed

def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    parsed = Path(path)
    try:
        return str(parsed.relative_to(REPO_ROOT))
    except ValueError:
        return str(parsed)

def _safe_float(value: Any, *, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required float value: {name}")
    return float(value)

def _safe_int(value: Any, *, name: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"Missing required integer value: {name}")
        return int(default)
    return int(value)

def _nested_value(payload: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = payload
        found = True
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found and current is not None:
            return current
    return None

def _file_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": _rel(path), "exists": False}
    return {
        "path": _rel(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }

def _load_membership(path: Path) -> np.ndarray:
    table = pq.read_table(path)
    names = set(table.column_names)
    if "cluster" not in names:
        raise ValueError(f"membership parquet must contain a cluster column: {path}")
    cluster = table.column("cluster").combine_chunks().to_numpy(zero_copy_only=False)
    if "node_idx" in names:
        node_idx = table.column("node_idx").combine_chunks().to_numpy(
            zero_copy_only=False
        )
        if node_idx.size and not np.all(node_idx[:-1] <= node_idx[1:]):
            cluster = cluster[np.argsort(node_idx, kind="stable")]
    return np.asarray(cluster, dtype=np.uint64)

def _membership_length(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return int(_load_membership(path).shape[0])

def _infer_n_nodes(input_cfg: Slice4Input) -> int:
    if input_cfg.n_nodes is not None:
        return int(input_cfg.n_nodes)
    if input_cfg.node_weights_path.exists():
        return int(
            input_cfg.node_weights_path.stat().st_size // np.dtype(np.float64).itemsize
        )
    membership_n = _membership_length(input_cfg.membership_path)
    if membership_n is not None:
        return int(membership_n)
    raise ValueError(
        "Could not infer n_nodes; provide --n-nodes or a node_weights.f64.bin sidecar"
    )

def _load_node_weights(path: Path, n_nodes: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing node weights sidecar: {path}")
    weights = np.fromfile(path, dtype=np.float64)
    if int(weights.shape[0]) != int(n_nodes):
        raise ValueError(
            f"node weight count mismatch: expected {n_nodes}, got {weights.shape[0]}"
        )
    return np.asarray(weights, dtype=np.float64)

def _resolve_input_from_summary(
    summary_path: Path,
    *,
    sample: str | None = None,
    resolution: float | None = None,
    target_max_doc_weight: float | None = None,
    seed: int | None = None,
) -> Slice4Input:
    summary = _read_json(summary_path)
    paths = summary.get("paths", {})
    graph_dir = _repo_path(
        paths.get("graph_dir")
        or paths.get("graph")
        or summary.get("graph_dir")
        or summary.get("graph")
    )
    membership_path = _repo_path(
        paths.get("membership")
        or paths.get("membership_path")
        or summary.get("membership")
        or summary.get("membership_path")
    )
    if graph_dir is None:
        raise ValueError(f"Missing graph_dir in {summary_path}")
    if membership_path is None:
        raise ValueError(f"Missing membership path in {summary_path}")
    node_weights_path = _repo_path(
        paths.get("node_weights")
        or paths.get("node_weights_path")
        or summary.get("node_weights")
        or summary.get("node_weights_path")
    )
    if node_weights_path is None:
        node_weights_path = graph_dir / "node_weights.f64.bin"
    return Slice4Input(
        sample=str(sample or summary.get("sample") or summary_path.parent.name),
        graph_dir=graph_dir,
        membership_path=membership_path,
        node_weights_path=node_weights_path,
        resolution=_safe_float(
            resolution
            if resolution is not None
            else _nested_value(summary, "resolution", "gamma", "config.resolution"),
            name="resolution",
        ),
        target_max_doc_weight=_safe_float(
            target_max_doc_weight
            if target_max_doc_weight is not None
            else _nested_value(
                summary,
                "target_max_doc_weight",
                "target_max_weight",
                "config.target_max_doc_weight",
            ),
            name="target_max_doc_weight",
        ),
        seed=_safe_int(
            seed
            if seed is not None
            else _nested_value(summary, "seed", "source_seed", "config.seed"),
            name="seed",
            default=42,
        ),
        n_nodes=(
            None
            if _nested_value(summary, "n_nodes", "config.n_nodes") is None
            else int(_nested_value(summary, "n_nodes", "config.n_nodes"))
        ),
        summary_path=summary_path,
    )

def _resolve_explicit_input(args: argparse.Namespace) -> Slice4Input:
    graph_dir = _repo_path(args.graph_dir)
    if graph_dir is None:
        raise ValueError("--graph-dir is required without --summary")
    node_weights_path = _repo_path(args.node_weights)
    if node_weights_path is None:
        node_weights_path = graph_dir / "node_weights.f64.bin"
    membership_path = _repo_path(args.membership)
    return Slice4Input(
        sample=str(args.sample or graph_dir.parent.name),
        graph_dir=graph_dir,
        membership_path=membership_path,
        node_weights_path=node_weights_path,
        resolution=_safe_float(args.resolution, name="resolution"),
        target_max_doc_weight=_safe_float(
            args.target_max_doc_weight,
            name="target_max_doc_weight",
        ),
        seed=_safe_int(args.seed, name="seed", default=42),
        n_nodes=None if args.n_nodes is None else int(args.n_nodes),
        summary_path=None,
    )

def _load_graph(input_cfg: Slice4Input, node_weights: np.ndarray) -> Any:
    required = [
        input_cfg.graph_dir / "src.u32.bin",
        input_cfg.graph_dir / "dst.u32.bin",
        input_cfg.graph_dir / "weight.f64.bin",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing graph sidecar files: "
            + ", ".join(str(path) for path in missing)
        )
    src = np.memmap(required[0], dtype=np.uint32, mode="r")
    dst = np.memmap(required[1], dtype=np.uint32, mode="r")
    weight = np.memmap(required[2], dtype=np.float64, mode="r")
    if src.shape[0] != dst.shape[0] or src.shape[0] != weight.shape[0]:
        raise ValueError(
            "graph sidecar length mismatch: "
            f"src={src.shape[0]} dst={dst.shape[0]} weight={weight.shape[0]}"
        )
    return build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=weight,
        n_nodes=int(node_weights.shape[0]),
        node_weights=np.asarray(node_weights, dtype=np.float64),
    )

def _cluster_weight_summary(
    membership: np.ndarray,
    node_weights: np.ndarray,
    target_max_doc_weight: float,
) -> dict[str, Any]:
    membership = np.asarray(membership, dtype=np.uint64)
    if int(membership.shape[0]) != int(node_weights.shape[0]):
        raise ValueError(
            "membership/node weight length mismatch: "
            f"{membership.shape[0]} vs {node_weights.shape[0]}"
        )
    if membership.size == 0:
        return {
            "max_doc_weight": 0.0,
            "max_doc_weight_ratio": 0.0,
            "n_above_max_doc_weight": 0,
            "top10_doc_weights": [],
        }
    _, inverse = np.unique(membership, return_inverse=True)
    weights = np.bincount(inverse, weights=np.asarray(node_weights, dtype=np.float64))
    sorted_desc = np.sort(weights)[::-1]
    max_weight = float(sorted_desc[0]) if sorted_desc.size else 0.0
    ratio = (
        float(max_weight / target_max_doc_weight)
        if target_max_doc_weight > 0.0
        else math.inf
    )
    return {
        "max_doc_weight": max_weight,
        "max_doc_weight_ratio": ratio,
        "n_above_max_doc_weight": int(np.count_nonzero(weights > target_max_doc_weight)),
        "top10_doc_weights": [float(x) for x in sorted_desc[:10]],
    }

def _membership_diff_summary(
    standard_membership: np.ndarray | None,
    membership: np.ndarray | None,
) -> dict[str, Any]:
    if standard_membership is None or membership is None:
        return {
            "membership_equal_to_standard": None,
            "membership_diff_nodes_vs_standard": None,
        }
    left = np.asarray(standard_membership, dtype=np.uint64)
    right = np.asarray(membership, dtype=np.uint64)
    if left.shape != right.shape:
        return {
            "membership_equal_to_standard": False,
            "membership_diff_nodes_vs_standard": None,
        }
    diff_nodes = int(np.count_nonzero(left != right))
    return {
        "membership_equal_to_standard": diff_nodes == 0,
        "membership_diff_nodes_vs_standard": diff_nodes,
    }

def _get_int_attr(obj: Any, name: str, default: int = 0) -> int:
    try:
        value = getattr(obj, name)
    except AttributeError:
        return int(default)
    if value is None:
        return int(default)
    return int(value)

def _get_float_attr(obj: Any, name: str, default: float = 0.0) -> float:
    try:
        value = getattr(obj, name)
    except AttributeError:
        return float(default)
    if value is None:
        return float(default)
    return float(value)

def _audit_metrics(result: Any | None) -> dict[str, Any]:
    audit = getattr(result, "audit", None)
    metrics: dict[str, Any] = {
        "audit_enabled": bool(getattr(audit, "enabled", False)),
        "n_iterations_used": _get_int_attr(result, "n_iterations_used", 0),
    }
    if audit is None:
        metrics["audit_iteration_count"] = 0
    else:
        iteration_depth = np.asarray(getattr(audit, "iteration_depth", []))
        metrics["audit_iteration_count"] = int(iteration_depth.shape[0])
    for field in AUDIT_INT_FIELDS:
        metrics[field] = _get_int_attr(audit, field, 0)
    for field in AUDIT_FLOAT_FIELDS:
        metrics[field] = _get_float_attr(audit, field, 0.0)
    return metrics

def _unsupported_row(*, sample: str, variant: str, reason: str) -> dict[str, Any]:
    row = {field: None for field in CSV_FIELDS}
    row.update(
        {
            "sample": sample,
            "variant": variant,
            "supported": False,
            "unsupported_reason": reason,
            "elapsed_sec": 0.0,
            "audit_enabled": False,
            "n_iterations_used": 0,
            "audit_iteration_count": 0,
        }
    )
    for field in AUDIT_INT_FIELDS:
        row[field] = 0
    for field in AUDIT_FLOAT_FIELDS:
        row[field] = 0.0
    return row

def _flatten_result(
    *,
    sample: str,
    variant: str,
    elapsed_sec: float,
    result: Any,
    node_weights: np.ndarray,
    target_max_doc_weight: float,
    standard_membership: np.ndarray | None = None,
    standard_quality: float | None = None,
) -> dict[str, Any]:
    membership = np.asarray(result.membership, dtype=np.uint64)
    size_summary = _cluster_weight_summary(
        membership,
        node_weights,
        target_max_doc_weight,
    )
    diff = _membership_diff_summary(standard_membership, membership)
    if variant == VARIANT_STANDARD:
        diff = {
            "membership_equal_to_standard": True,
            "membership_diff_nodes_vs_standard": 0,
        }
    quality = float(result.quality)
    quality_delta = (
        0.0
        if standard_quality is None
        else float(quality - float(standard_quality))
    )
    row = {
        "sample": sample,
        "variant": variant,
        "supported": True,
        "unsupported_reason": "",
        "elapsed_sec": float(elapsed_sec),
        "n_clusters": int(result.n_clusters),
        "quality": quality,
        "quality_delta_vs_standard": quality_delta,
        "quality_delta_vs_repair_off": None,
        "quality_improved_vs_standard": quality_delta > 0.0,
        "quality_improved_vs_repair_off": None,
        "membership_equal_repair_off_on": None,
        **size_summary,
        **diff,
        **_audit_metrics(result if variant != VARIANT_STANDARD else None),
    }
    return {field: row.get(field) for field in CSV_FIELDS}

def _run_variant(
    *,
    graph: Any,
    input_cfg: Slice4Input,
    run_config: Slice4RunConfig,
    node_weights: np.ndarray,
    variant: str,
    standard_membership: np.ndarray | None,
    standard_quality: float | None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    start = time.perf_counter()
    try:
        if variant == VARIANT_STANDARD:
            result = graph.run_leiden(
                resolution=float(input_cfg.resolution),
                seed=int(input_cfg.seed),
                n_iterations=int(run_config.n_iterations),
                randomness=float(run_config.randomness),
            )
        else:
            result = graph.run_leiden_dongdaemun_refinement(
                target_max_weight=float(input_cfg.target_max_doc_weight),
                resolution=float(input_cfg.resolution),
                seed=int(input_cfg.seed),
                n_iterations=int(run_config.n_iterations),
                randomness=float(run_config.randomness),
                soft_min_ratio=float(run_config.soft_min_ratio),
                max_extra_parents_per_iteration=int(
                    run_config.max_extra_parents_per_iteration
                ),
                max_extra_children_per_parent=int(
                    run_config.max_extra_children_per_parent
                ),
                parent_selection_policy=str(run_config.parent_selection_policy),
                max_singleton_weight_fraction=float(
                    run_config.max_singleton_weight_fraction
                ),
                min_largest_child_fraction_improvement=float(
                    run_config.min_largest_child_fraction_improvement
                ),
                gamma_multipliers=tuple(float(x) for x in run_config.gamma_multipliers),
                seed_perturbations=int(run_config.seed_perturbations),
                use_quotient_diagnostic=bool(run_config.use_quotient_diagnostic),
                use_baseline_repair=(variant == VARIANT_REPAIR_ON),
                baseline_repair_policy=str(run_config.baseline_repair_policy),
                baseline_repair_replace_min_parent_ratio=float(
                    run_config.baseline_repair_replace_min_parent_ratio
                ),
                baseline_repair_epsilon=float(run_config.baseline_repair_epsilon),
                candidate_quality_policy=str(run_config.candidate_quality_policy),
                min_candidate_delta_q=float(run_config.min_candidate_delta_q),
                adaptive_plateau_quality_band=float(
                    run_config.adaptive_plateau_quality_band
                ),
                use_final_quality_guard=bool(run_config.use_final_quality_guard),
                min_final_quality_delta=float(run_config.min_final_quality_delta),
                adaptive_probe_mode=str(run_config.adaptive_probe_mode),
                adaptive_probe_perturbations=int(run_config.adaptive_probe_perturbations),
                adaptive_probe_targets=tuple(run_config.adaptive_probe_targets),
                adaptive_probe_tolerance_parent_weight=float(
                    run_config.adaptive_probe_tolerance_parent_weight
                ),
                adaptive_probe_include_node_order_control=bool(
                    run_config.adaptive_probe_include_node_order_control
                ),
                adaptive_probe_commit_min_gain_parent_weight=float(
                    run_config.adaptive_probe_commit_min_gain_parent_weight
                ),
                adaptive_probe_max_commits_total=int(
                    run_config.adaptive_probe_max_commits_total
                ),
                adaptive_probe_max_commits_per_depth=int(
                    run_config.adaptive_probe_max_commits_per_depth
                ),
                adaptive_probe_commit_sources=tuple(
                    run_config.adaptive_probe_commit_sources
                ),
                adaptive_probe_commit_strategy=str(
                    run_config.adaptive_probe_commit_strategy
                ),
                adaptive_near_tie_probe_mode=str(
                    run_config.adaptive_near_tie_probe_mode
                ),
                adaptive_near_tie_margin_parent_weight=float(
                    run_config.adaptive_near_tie_margin_parent_weight
                ),
                adaptive_near_tie_randomness=float(
                    run_config.adaptive_near_tie_randomness
                ),
                adaptive_near_tie_max_decisions_per_parent=int(
                    run_config.adaptive_near_tie_max_decisions_per_parent
                ),
                adaptive_local_shake_mode=str(run_config.adaptive_local_shake_mode),
                adaptive_local_shake_arms=tuple(run_config.adaptive_local_shake_arms),
                adaptive_local_shake_max_arms_per_parent=int(
                    run_config.adaptive_local_shake_max_arms_per_parent
                ),
                adaptive_local_shake_max_candidates_per_parent=int(
                    run_config.adaptive_local_shake_max_candidates_per_parent
                ),
                adaptive_local_shake_min_gain_parent_weight=float(
                    run_config.adaptive_local_shake_min_gain_parent_weight
                ),
                adaptive_local_shake_shape_eps=float(
                    run_config.adaptive_local_shake_shape_eps
                ),
                adaptive_local_shake_arm_priority=tuple(
                    run_config.adaptive_local_shake_arm_priority
                ),
                adaptive_local_shake_near_tie_min_count=int(
                    run_config.adaptive_local_shake_near_tie_min_count
                ),
                adaptive_local_shake_resolution_down_multipliers=tuple(
                    float(x)
                    for x in run_config.adaptive_local_shake_resolution_down_multipliers
                ),
                adaptive_local_shake_resolution_up_multipliers=tuple(
                    float(x)
                    for x in run_config.adaptive_local_shake_resolution_up_multipliers
                ),
                adaptive_local_shake_resolution_up_min_parent_ratio=float(
                    run_config.adaptive_local_shake_resolution_up_min_parent_ratio
                ),
                adaptive_local_shake_resolution_down_max_parent_ratio=float(
                    run_config.adaptive_local_shake_resolution_down_max_parent_ratio
                ),
                adaptive_local_shake_large_child_fraction=float(
                    run_config.adaptive_local_shake_large_child_fraction
                ),
                adaptive_local_shake_singleton_fraction=float(
                    run_config.adaptive_local_shake_singleton_fraction
                ),
                adaptive_local_shake_seed_perturbations=int(
                    run_config.adaptive_local_shake_seed_perturbations
                ),
                adaptive_local_shake_seed_margin_count=int(
                    run_config.adaptive_local_shake_seed_margin_count
                ),
                adaptive_local_shake_near_tie_margin_parent_weight=float(
                    run_config.adaptive_local_shake_near_tie_margin_parent_weight
                ),
                adaptive_local_shake_near_tie_randomness=float(
                    run_config.adaptive_local_shake_near_tie_randomness
                ),
                adaptive_local_shake_final_guard_mode=str(
                    run_config.adaptive_local_shake_final_guard_mode
                ),
            )
    except (AttributeError, TypeError, ImportError) as exc:
        if variant == VARIANT_STANDARD:
            raise
        return (
            _unsupported_row(
                sample=input_cfg.sample,
                variant=variant,
                reason=str(exc),
            ),
            None,
        )
    elapsed = time.perf_counter() - start
    row = _flatten_result(
        sample=input_cfg.sample,
        variant=variant,
        elapsed_sec=elapsed,
        result=result,
        node_weights=node_weights,
        target_max_doc_weight=float(input_cfg.target_max_doc_weight),
        standard_membership=standard_membership,
        standard_quality=standard_quality,
    )
    return row, np.asarray(result.membership, dtype=np.uint64)

def _set_cross_variant_comparisons(
    rows: list[dict[str, Any]],
    repair_off: np.ndarray | None,
    repair_on: np.ndarray | None,
) -> None:
    if repair_off is None or repair_on is None or repair_off.shape != repair_on.shape:
        equal: bool | None = None if repair_off is None or repair_on is None else False
    else:
        equal = bool(np.array_equal(repair_off, repair_on))
    for row in rows:
        if row.get("variant") in {VARIANT_REPAIR_OFF, VARIANT_REPAIR_ON}:
            row["membership_equal_repair_off_on"] = equal
    by_variant = {str(row.get("variant")): row for row in rows}
    repair_off_row = by_variant.get(VARIANT_REPAIR_OFF)
    if repair_off_row is None or repair_off_row.get("quality") is None:
        return
    repair_off_quality = float(repair_off_row["quality"])
    for row in rows:
        if row.get("quality") is None:
            continue
        delta = float(row["quality"]) - repair_off_quality
        row["quality_delta_vs_repair_off"] = delta
        row["quality_improved_vs_repair_off"] = delta > 0.0

def _sum_int(rows: list[dict[str, Any]], field: str) -> int:
    return int(sum(int(row.get(field) or 0) for row in rows))

def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)

def _audit_profile_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    same_candidates = _sum_int(rows, "same_gamma_candidates_total")
    high_candidates = _sum_int(rows, "high_gamma_candidates_total")
    return {
        "quadrants": {
            "qpos_spos": _sum_int(rows, "candidate_qpos_spos_total"),
            "qpos_sneg": _sum_int(rows, "candidate_qpos_sneg_total"),
            "qneg_spos": _sum_int(rows, "candidate_qneg_spos_total"),
            "qneg_sneg": _sum_int(rows, "candidate_qneg_sneg_total"),
        },
        "decision_confusion": {
            "true_positive": _sum_int(rows, "candidate_true_positive_total"),
            "false_positive": _sum_int(rows, "candidate_false_positive_total"),
            "false_negative": _sum_int(rows, "candidate_false_negative_total"),
            "true_negative": _sum_int(rows, "candidate_true_negative_total"),
        },
        "candidate_validity": {
            "valid": _sum_int(rows, "candidate_valid_total"),
            "invalid": _sum_int(rows, "candidate_invalid_total"),
            "quality_rejected": _sum_int(rows, "candidate_rejected_by_quality_total"),
            "policy_rejected": _sum_int(rows, "candidate_rejected_by_policy_total"),
        },
        "same_gamma": {
            "candidates": same_candidates,
            "qpos_spos": _sum_int(rows, "same_gamma_qpos_spos_total"),
            "qpos_spos_rate": _rate(
                _sum_int(rows, "same_gamma_qpos_spos_total"),
                same_candidates,
            ),
            "applied": _sum_int(rows, "same_gamma_applied_total"),
            "applied_rate": _rate(
                _sum_int(rows, "same_gamma_applied_total"),
                same_candidates,
            ),
        },
        "high_gamma": {
            "candidates": high_candidates,
            "qpos_spos": _sum_int(rows, "high_gamma_qpos_spos_total"),
            "qpos_spos_rate": _rate(
                _sum_int(rows, "high_gamma_qpos_spos_total"),
                high_candidates,
            ),
            "applied": _sum_int(rows, "high_gamma_applied_total"),
            "applied_rate": _rate(
                _sum_int(rows, "high_gamma_applied_total"),
                high_candidates,
            ),
        },
    }

def _aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {str(row.get("variant")): row for row in rows}
    repair_off = by_variant.get(VARIANT_REPAIR_OFF, {})
    repair_on = by_variant.get(VARIANT_REPAIR_ON, {})
    repair_off_zero = all(
        float(repair_off.get(field) or 0.0) == 0.0
        for field in BASELINE_REPAIR_AUDIT_FIELDS
    )
    if not repair_on.get("supported", False):
        repair_on_status = "unsupported"
    elif int(repair_on.get("baseline_repair_candidates_total") or 0) > 0:
        repair_on_status = "repair exercised"
    else:
        repair_on_status = "no repair opportunity"
    return {
        "n_rows": int(len(rows)),
        "rust_dongdaemun_refinement_available": bool(
            RUST_DONGDAEMUN_REFINEMENT_AVAILABLE
        ),
        "repair_off_baseline_repair_audit_zero": bool(repair_off_zero),
        "repair_on_repair_status": repair_on_status,
        "membership_equal_repair_off_on": repair_on.get(
            "membership_equal_repair_off_on"
        ),
        "quality_improved_vs_standard_variants": [
            variant
            for variant, row in by_variant.items()
            if bool(row.get("quality_improved_vs_standard", False))
        ],
        "repair_on_quality_delta_vs_repair_off": repair_on.get(
            "quality_delta_vs_repair_off"
        ),
        "repair_on_quality_improved_vs_repair_off": repair_on.get(
            "quality_improved_vs_repair_off"
        ),
        "baseline_repair_audit_semantics": (
            "baseline_repair_merge_count_total sums internal merges across "
            "evaluated repair candidates; baseline_repair_selected_total counts "
            "final selected parent candidates whose selected candidate included "
            "repair merges."
        ),
        "candidate_profile": _audit_profile_summary(rows),
        "quality_delta_vs_standard": {
            variant: row.get("quality_delta_vs_standard")
            for variant, row in by_variant.items()
        },
        "max_doc_weight": {
            variant: row.get("max_doc_weight") for variant, row in by_variant.items()
        },
        "n_above_max_doc_weight": {
            variant: row.get("n_above_max_doc_weight")
            for variant, row in by_variant.items()
        },
    }

def _csv_value(value: Any) -> Any:
    safe = _json_safe(value)
    if safe is None:
        return ""
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return safe

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})

def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [{field: _json_safe(row.get(field)) for field in CSV_FIELDS} for row in rows]
    )
    pq.write_table(table, path)

def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| variant | supported | quality delta | max doc weight | n above target | repair candidates | elapsed sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {supported} | {quality_delta:.6g} | {max_weight:.6g} | {n_above} | {repair_candidates} | {elapsed:.3f} |".format(
                variant=row.get("variant", ""),
                supported=bool(row.get("supported", False)),
                quality_delta=float(row.get("quality_delta_vs_standard") or 0.0),
                max_weight=float(row.get("max_doc_weight") or 0.0),
                n_above=int(row.get("n_above_max_doc_weight") or 0),
                repair_candidates=int(
                    row.get("baseline_repair_candidates_total") or 0
                ),
                elapsed=float(row.get("elapsed_sec") or 0.0),
            )
        )
    return lines

def _write_report(
    path: Path,
    *,
    input_cfg: Slice4Input,
    run_config: Slice4RunConfig,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = aggregate.get("candidate_profile") or {}
    quadrants = profile.get("quadrants") or {}
    confusion = profile.get("decision_confusion") or {}
    same_gamma = profile.get("same_gamma") or {}
    high_gamma = profile.get("high_gamma") or {}
    lines = [
        "# Dongdaemun Refinement Slice 4 Pilot",
        "",
        "This pilot compares standard Leiden with integrated Dongdaemun refinement on the same prepared source-level graph, resolution, and seed.",
        "",
        "## Input",
        "",
        f"- Sample: {input_cfg.sample}",
        f"- Graph dir: `{_rel(input_cfg.graph_dir)}`",
        f"- Membership: `{_rel(input_cfg.membership_path)}`",
        f"- Node weights: `{_rel(input_cfg.node_weights_path)}`",
        f"- Resolution: {input_cfg.resolution:g}",
        f"- Target max doc weight: {input_cfg.target_max_doc_weight:g}",
        f"- Seed: {input_cfg.seed}",
        "",
        "## Variant Rows",
        "",
        *_markdown_table(rows),
        "",
        "## Safety Checks",
        "",
        f"- Repair-off baseline repair audit is zero: {aggregate.get('repair_off_baseline_repair_audit_zero')}",
        f"- Repair-on status: {aggregate.get('repair_on_repair_status')}",
        f"- Repair off/on membership equal: {aggregate.get('membership_equal_repair_off_on')}",
        f"- Refinement variants above standard quality: {aggregate.get('quality_improved_vs_standard_variants')}",
        f"- Repair-on quality delta vs repair-off: {aggregate.get('repair_on_quality_delta_vs_repair_off')}",
        f"- Baseline repair audit semantics: {aggregate.get('baseline_repair_audit_semantics')}",
        "",
        "## Candidate Profile",
        "",
        "- Q/S quadrants: "
        f"Q+/S+={quadrants.get('qpos_spos', 0)}, "
        f"Q+/S-={quadrants.get('qpos_sneg', 0)}, "
        f"Q-/S+={quadrants.get('qneg_spos', 0)}, "
        f"Q-/S-={quadrants.get('qneg_sneg', 0)}.",
        "- Decision confusion: "
        f"TP={confusion.get('true_positive', 0)}, "
        f"FP={confusion.get('false_positive', 0)}, "
        f"FN={confusion.get('false_negative', 0)}, "
        f"TN={confusion.get('true_negative', 0)}.",
        "- Source success rates: "
        f"same-gamma Q+/S+={same_gamma.get('qpos_spos_rate')}, "
        f"high-gamma Q+/S+={high_gamma.get('qpos_spos_rate')}; "
        f"same-gamma applied={same_gamma.get('applied_rate')}, "
        f"high-gamma applied={high_gamma.get('applied_rate')}.",
        "",
        "## Config",
        "",
        f"- n_iterations: {run_config.n_iterations}",
        f"- randomness: {run_config.randomness:g}",
        f"- soft_min_ratio: {run_config.soft_min_ratio:g}",
        f"- gamma_multipliers: {', '.join(f'{x:g}' for x in run_config.gamma_multipliers)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

def _write_outputs(
    *,
    output_dir: Path,
    input_cfg: Slice4Input,
    run_config: Slice4RunConfig,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "slice4_refinement_pilot.csv"
    parquet_path = output_dir / "slice4_refinement_pilot.parquet"
    summary_path = output_dir / "slice4_refinement_pilot_summary.json"
    report_path = output_dir / "slice4_refinement_pilot_report.md"
    _write_csv(csv_path, rows)
    _write_parquet(parquet_path, rows)
    payload = {
        "schema": f"dongdaemun_refinement_slice4_pilot.v{SCHEMA_VERSION}",
        "input": asdict(input_cfg),
        "config": asdict(run_config),
        "input_fingerprint": {
            "summary": _file_fingerprint(input_cfg.summary_path),
            "membership": _file_fingerprint(input_cfg.membership_path),
            "node_weights": _file_fingerprint(input_cfg.node_weights_path),
            "graph_src": _file_fingerprint(input_cfg.graph_dir / "src.u32.bin"),
            "graph_dst": _file_fingerprint(input_cfg.graph_dir / "dst.u32.bin"),
            "graph_weight": _file_fingerprint(input_cfg.graph_dir / "weight.f64.bin"),
        },
        "aggregate": aggregate,
        "rows": rows,
        "paths": {
            "csv": csv_path,
            "parquet": parquet_path,
            "summary": summary_path,
            "report": report_path,
        },
    }
    _write_json(summary_path, payload)
    _write_report(
        report_path,
        input_cfg=input_cfg,
        run_config=run_config,
        rows=rows,
        aggregate=aggregate,
    )
    return {
        "csv": csv_path,
        "parquet": parquet_path,
        "summary": summary_path,
        "report": report_path,
    }

def run_pilot(
    input_cfg: Slice4Input,
    *,
    output_dir: Path,
    run_config: Slice4RunConfig | None = None,
) -> dict[str, Any]:
    run_config = run_config or Slice4RunConfig()
    n_nodes = _infer_n_nodes(input_cfg)
    node_weights = _load_node_weights(input_cfg.node_weights_path, n_nodes)
    graph = _load_graph(input_cfg, node_weights)

    standard_row, standard_membership = _run_variant(
        graph=graph,
        input_cfg=input_cfg,
        run_config=run_config,
        node_weights=node_weights,
        variant=VARIANT_STANDARD,
        standard_membership=None,
        standard_quality=None,
    )
    standard_quality = float(standard_row["quality"])
    repair_off_row, repair_off_membership = _run_variant(
        graph=graph,
        input_cfg=input_cfg,
        run_config=run_config,
        node_weights=node_weights,
        variant=VARIANT_REPAIR_OFF,
        standard_membership=standard_membership,
        standard_quality=standard_quality,
    )
    repair_on_row, repair_on_membership = _run_variant(
        graph=graph,
        input_cfg=input_cfg,
        run_config=run_config,
        node_weights=node_weights,
        variant=VARIANT_REPAIR_ON,
        standard_membership=standard_membership,
        standard_quality=standard_quality,
    )
    rows = [standard_row, repair_off_row, repair_on_row]
    _set_cross_variant_comparisons(rows, repair_off_membership, repair_on_membership)
    aggregate = _aggregate_summary(rows)
    paths = _write_outputs(
        output_dir=output_dir,
        input_cfg=input_cfg,
        run_config=run_config,
        rows=rows,
        aggregate=aggregate,
    )
    return {
        "input": input_cfg,
        "config": run_config,
        "rows": rows,
        "aggregate": aggregate,
        "paths": paths,
    }

def _parse_float_tuple(value: str | None) -> tuple[float, ...]:
    if value is None:
        return DEFAULT_DONGDAEMUN_GAMMA_MULTIPLIERS
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())

def _parse_optional_float_tuple(value: str | None) -> tuple[float, ...]:
    if value is None:
        return ()
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())

def _parse_str_tuple(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", "--prepare-summary", dest="summary", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--node-weights", type=Path)
    parser.add_argument("--n-nodes", type=int)
    parser.add_argument("--sample")
    parser.add_argument("--resolution", type=float)
    parser.add_argument("--target-max-doc-weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--soft-min-ratio", type=float, default=1.0)
    parser.add_argument("--max-extra-parents-per-iteration", type=int, default=16)
    parser.add_argument("--max-extra-children-per-parent", type=int, default=64)
    parser.add_argument(
        "--parent-selection-policy",
        choices=("weight", "pressure_boundary"),
        default="weight",
    )
    parser.add_argument("--max-singleton-weight-fraction", type=float, default=0.05)
    parser.add_argument(
        "--min-largest-child-fraction-improvement",
        type=float,
        default=0.05,
    )
    parser.add_argument("--gamma-multipliers")
    parser.add_argument("--seed-perturbations", type=int, default=0)
    parser.add_argument(
        "--candidate-quality-policy",
        choices=(
            "structural",
            "quality_guarded_structural",
            "quality_floor",
            "quality_first",
            "selective",
            "pressure_aware",
            "adaptive_plateau",
        ),
        default="structural",
    )
    parser.add_argument("--min-candidate-delta-q", type=float, default=0.0)
    parser.add_argument("--adaptive-plateau-quality-band", type=float, default=0.0)
    parser.add_argument("--use-final-quality-guard", action="store_true")
    parser.add_argument("--min-final-quality-delta", type=float, default=0.0)
    parser.add_argument(
        "--baseline-repair-policy",
        choices=("replace", "augment", "adaptive"),
        default="replace",
    )
    parser.add_argument(
        "--baseline-repair-replace-min-parent-ratio",
        type=float,
        default=1.05,
    )
    parser.add_argument(
        "--adaptive-near-tie-probe-mode",
        choices=("off", "trace_only", "candidate", "qf_replace", "near_tie_qf_replace"),
        default="off",
    )
    parser.add_argument(
        "--adaptive-near-tie-margin-parent-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--adaptive-near-tie-randomness", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-near-tie-max-decisions-per-parent",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--adaptive-local-shake-mode",
        choices=("off", "trace_only", "qf_replace", "pressure_guarded"),
        default="off",
    )
    parser.add_argument("--adaptive-local-shake-arms")
    parser.add_argument("--adaptive-local-shake-max-arms-per-parent", type=int, default=0)
    parser.add_argument("--adaptive-local-shake-max-candidates-per-parent", type=int, default=0)
    parser.add_argument("--adaptive-local-shake-min-gain-parent-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-local-shake-shape-eps", type=float, default=1e-12)
    parser.add_argument("--adaptive-local-shake-arm-priority")
    parser.add_argument("--adaptive-local-shake-near-tie-min-count", type=int, default=1)
    parser.add_argument("--adaptive-local-shake-resolution-down-multipliers")
    parser.add_argument("--adaptive-local-shake-resolution-up-multipliers")
    parser.add_argument("--adaptive-local-shake-resolution-up-min-parent-ratio", type=float, default=1.0)
    parser.add_argument("--adaptive-local-shake-resolution-down-max-parent-ratio", type=float, default=1.0)
    parser.add_argument("--adaptive-local-shake-large-child-fraction", type=float, default=0.95)
    parser.add_argument("--adaptive-local-shake-singleton-fraction", type=float, default=0.05)
    parser.add_argument("--adaptive-local-shake-seed-perturbations", type=int, default=0)
    parser.add_argument("--adaptive-local-shake-seed-margin-count", type=int, default=2)
    parser.add_argument("--adaptive-local-shake-near-tie-margin-parent-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-local-shake-near-tie-randomness", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-local-shake-final-guard-mode",
        choices=("none", "runner_audit"),
        default="none",
    )
    return parser

def _input_from_args(args: argparse.Namespace) -> Slice4Input:
    if args.summary is not None:
        return _resolve_input_from_summary(
            args.summary,
            sample=args.sample,
            resolution=args.resolution,
            target_max_doc_weight=args.target_max_doc_weight,
            seed=args.seed,
        )
    return _resolve_explicit_input(args)

def _run_config_from_args(args: argparse.Namespace) -> Slice4RunConfig:
    return Slice4RunConfig(
        n_iterations=int(args.n_iterations),
        randomness=float(args.randomness),
        soft_min_ratio=float(args.soft_min_ratio),
        max_extra_parents_per_iteration=int(args.max_extra_parents_per_iteration),
        max_extra_children_per_parent=int(args.max_extra_children_per_parent),
        parent_selection_policy=str(args.parent_selection_policy),
        max_singleton_weight_fraction=float(args.max_singleton_weight_fraction),
        min_largest_child_fraction_improvement=float(
            args.min_largest_child_fraction_improvement
        ),
        gamma_multipliers=_parse_float_tuple(args.gamma_multipliers),
        seed_perturbations=int(args.seed_perturbations),
        use_quotient_diagnostic=True,
        baseline_repair_policy=str(args.baseline_repair_policy),
        baseline_repair_replace_min_parent_ratio=float(
            args.baseline_repair_replace_min_parent_ratio
        ),
        baseline_repair_epsilon=0.0,
        candidate_quality_policy=str(args.candidate_quality_policy),
        min_candidate_delta_q=float(args.min_candidate_delta_q),
        adaptive_plateau_quality_band=float(args.adaptive_plateau_quality_band),
        use_final_quality_guard=bool(args.use_final_quality_guard),
        min_final_quality_delta=float(args.min_final_quality_delta),
        adaptive_near_tie_probe_mode=str(args.adaptive_near_tie_probe_mode),
        adaptive_near_tie_margin_parent_weight=float(
            args.adaptive_near_tie_margin_parent_weight
        ),
        adaptive_near_tie_randomness=float(args.adaptive_near_tie_randomness),
        adaptive_near_tie_max_decisions_per_parent=int(
            args.adaptive_near_tie_max_decisions_per_parent
        ),
        adaptive_local_shake_mode=str(args.adaptive_local_shake_mode),
        adaptive_local_shake_arms=_parse_str_tuple(args.adaptive_local_shake_arms),
        adaptive_local_shake_max_arms_per_parent=int(
            args.adaptive_local_shake_max_arms_per_parent
        ),
        adaptive_local_shake_max_candidates_per_parent=int(
            args.adaptive_local_shake_max_candidates_per_parent
        ),
        adaptive_local_shake_min_gain_parent_weight=float(
            args.adaptive_local_shake_min_gain_parent_weight
        ),
        adaptive_local_shake_shape_eps=float(args.adaptive_local_shake_shape_eps),
        adaptive_local_shake_arm_priority=_parse_str_tuple(
            args.adaptive_local_shake_arm_priority
        ),
        adaptive_local_shake_near_tie_min_count=int(
            args.adaptive_local_shake_near_tie_min_count
        ),
        adaptive_local_shake_resolution_down_multipliers=_parse_optional_float_tuple(
            args.adaptive_local_shake_resolution_down_multipliers
        ),
        adaptive_local_shake_resolution_up_multipliers=_parse_optional_float_tuple(
            args.adaptive_local_shake_resolution_up_multipliers
        ),
        adaptive_local_shake_resolution_up_min_parent_ratio=float(
            args.adaptive_local_shake_resolution_up_min_parent_ratio
        ),
        adaptive_local_shake_resolution_down_max_parent_ratio=float(
            args.adaptive_local_shake_resolution_down_max_parent_ratio
        ),
        adaptive_local_shake_large_child_fraction=float(
            args.adaptive_local_shake_large_child_fraction
        ),
        adaptive_local_shake_singleton_fraction=float(
            args.adaptive_local_shake_singleton_fraction
        ),
        adaptive_local_shake_seed_perturbations=int(
            args.adaptive_local_shake_seed_perturbations
        ),
        adaptive_local_shake_seed_margin_count=int(
            args.adaptive_local_shake_seed_margin_count
        ),
        adaptive_local_shake_near_tie_margin_parent_weight=float(
            args.adaptive_local_shake_near_tie_margin_parent_weight
        ),
        adaptive_local_shake_near_tie_randomness=float(
            args.adaptive_local_shake_near_tie_randomness
        ),
        adaptive_local_shake_final_guard_mode=str(
            args.adaptive_local_shake_final_guard_mode
        ),
    )

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    input_cfg = _input_from_args(args)
    payload = run_pilot(
        input_cfg,
        output_dir=args.output_dir,
        run_config=_run_config_from_args(args),
    )
    print(f"Saved Slice 4 pilot outputs to {_rel(payload['paths']['summary'])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
