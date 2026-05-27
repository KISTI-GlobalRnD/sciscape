#!/usr/bin/env python3
"""Collect same-case vanilla Leiden basin signatures for reachability audits.

This runner is intentionally narrower than the random-refinement profiler.  It
uses the graph dirs from an existing portfolio batch manifest, runs only the
standard Rust Leiden path, and writes `vanilla_basin_rows.csv` in the schema
expected by `analyze_leiden_basin_reachability_audit.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from run_leiden_hysteresis_shatter_smoke import _case_name, _load_graph_arrays  # noqa: E402
from run_leiden_hysteresis_work_acceleration_monitor import (  # noqa: E402
    _reconstruct_external_group,
)
from sciscape.clustering.leiden_rust import build_leiden_graph  # noqa: E402


DEFAULT_CASE_MANIFEST = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/portfolio_batch_cases.csv"
)
DEFAULT_TARGET_ROWS = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "basin_reachability_audit/dongdaemun_basin_reachability_target_rows.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "vanilla_reachability_sweep"
)
DEFAULT_SEEDS = (11, 42, 73, 101, 137)
DEFAULT_RANDOMNESS = (0.0, 0.001, 0.01)
DEFAULT_N_ITERATIONS_VALUES = ("1", "10", "convergence")
VARIANT_STANDARD = "standard_leiden"
ROWS_FILENAME = "vanilla_basin_rows.csv"
SUMMARY_FILENAME = "vanilla_basin_sweep_summary.json"
REPORT_FILENAME = "vanilla_basin_sweep_report.md"
BASIN_SIGNATURE_SKETCH_SAMPLE_SIZE = 1024
BASIN_CHANGED_SUPPORT_SKETCH_SAMPLE_SIZE = 8192
SKETCH_HASH_COLUMN = "p5_basin_sketch_node_hash"
SKETCH_BASELINE_COLUMN = "p5_basin_sketch_baseline_membership"
SKETCH_MEMBERSHIP_COLUMN = "p5_basin_sketch_membership"
CHANGED_SUPPORT_COLUMN = "p5_basin_changed_support_nodes"
ALIGNMENT_ERROR_NODE_COUNT_COLUMN = "p5_alignment_error_nodes_vs_baseline"
ALIGNMENT_ERROR_FRACTION_COLUMN = "p5_alignment_error_fraction_vs_baseline"
ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN = "p5_aligned_changed_support_node_count"
ALIGNED_CHANGED_SUPPORT_NODES_COLUMN = "p5_aligned_changed_support_nodes"


@dataclass(frozen=True)
class IterationBudget:
    requested: str
    n_iterations: int
    mode: str


def _parse_csv(value: str | None, *, cast: type = str) -> tuple[Any, ...]:
    if value is None:
        return ()
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(cast(item) for item in items)


def _parse_n_iterations_value(value: Any) -> IterationBudget:
    token = str(value).strip().lower()
    if token in {"convergence", "converge", "until_convergence", "0"}:
        return IterationBudget("convergence", 0, "convergence")
    try:
        n_iterations = int(token)
    except ValueError as exc:
        raise ValueError(
            f"Invalid n_iterations value {value!r}; use a positive integer or convergence"
        ) from exc
    if n_iterations <= 0:
        raise ValueError(
            f"Invalid n_iterations value {value!r}; use a positive integer or convergence"
        )
    return IterationBudget(str(n_iterations), n_iterations, "fixed")


def _parse_n_iterations_values(value: str) -> tuple[IterationBudget, ...]:
    budgets = tuple(
        _parse_n_iterations_value(part)
        for part in value.split(",")
        if part.strip()
    )
    if not budgets:
        raise ValueError("--n-iterations-values must contain at least one value")
    seen: set[str] = set()
    for budget in budgets:
        if budget.requested in seen:
            raise ValueError(f"Duplicate n_iterations value: {budget.requested}")
        seen.add(budget.requested)
    return budgets


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _mix_fnv1a_u64(hash_value: int, value: int) -> int:
    prime = 1_099_511_628_211
    mask = (1 << 64) - 1
    for byte in int(value).to_bytes(8, byteorder="little", signed=False):
        hash_value ^= byte
        hash_value = (hash_value * prime) & mask
    return hash_value


def stable_node_sample_key(node: int) -> int:
    mask = (1 << 64) - 1
    x = (int(node) + 0x9E37_79B9_7F4A_7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58_476D_1CE4_E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D0_49BB_1331_11EB) & mask
    return (x ^ (x >> 31)) & mask


def stable_sample_nodes(nodes: np.ndarray, max_nodes: int) -> np.ndarray:
    unique = np.unique(np.asarray(nodes, dtype=np.uint32))
    if max_nodes <= 0:
        return np.asarray([], dtype=np.uint32)
    if unique.shape[0] > max_nodes:
        ordered = sorted(
            (int(node) for node in unique),
            key=lambda node: (stable_node_sample_key(node), node),
        )
        unique = np.asarray(sorted(ordered[:max_nodes]), dtype=np.uint32)
    return np.asarray(unique, dtype=np.uint32)


def hash_u32_sequence(values: np.ndarray) -> str:
    hash_value = 14_695_981_039_346_656_037
    values = np.asarray(values, dtype=np.uint32)
    hash_value = _mix_fnv1a_u64(hash_value, int(values.shape[0]))
    for value in values:
        hash_value = _mix_fnv1a_u64(hash_value, int(value))
    return f"{hash_value:016x}"


def encode_u32_sequence(values: np.ndarray) -> str:
    return ";".join(str(int(value)) for value in np.asarray(values, dtype=np.uint32))


def encode_membership_sketch(labels: np.ndarray, nodes: np.ndarray) -> str:
    labels = np.asarray(labels, dtype=np.uint64)
    return ";".join(
        str(int(labels[int(node)])) if int(node) < labels.shape[0] else str(2**32 - 1)
        for node in np.asarray(nodes, dtype=np.uint32)
    )


def canonical_partition_signature(membership: np.ndarray) -> tuple[str, int]:
    labels = np.asarray(membership, dtype=np.uint64)
    canonical_ids: dict[int, int] = {}
    next_id = 0
    hash_value = 14_695_981_039_346_656_037
    hash_value = _mix_fnv1a_u64(hash_value, int(labels.shape[0]))
    for label in labels:
        key = int(label)
        if key not in canonical_ids:
            canonical_ids[key] = next_id
            next_id += 1
        hash_value = _mix_fnv1a_u64(hash_value, canonical_ids[key])
    return f"{hash_value:016x}", next_id


def _best_partner_maps(
    baseline: np.ndarray,
    membership: np.ndarray,
) -> tuple[dict[int, int], dict[int, int]]:
    pair_counts: dict[tuple[int, int], int] = {}
    for left, right in zip(baseline, membership, strict=False):
        key = (int(left), int(right))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    baseline_best: dict[int, tuple[int, int]] = {}
    membership_best: dict[int, tuple[int, int]] = {}
    for (baseline_cluster, membership_cluster), count in pair_counts.items():
        current = baseline_best.get(baseline_cluster)
        if current is None or count > current[1] or (
            count == current[1] and membership_cluster < current[0]
        ):
            baseline_best[baseline_cluster] = (membership_cluster, count)
        current = membership_best.get(membership_cluster)
        if current is None or count > current[1] or (
            count == current[1] and baseline_cluster < current[0]
        ):
            membership_best[membership_cluster] = (baseline_cluster, count)
    return (
        {cluster: partner for cluster, (partner, _count) in baseline_best.items()},
        {cluster: partner for cluster, (partner, _count) in membership_best.items()},
    )


def basin_sketch_stats(
    *,
    baseline: np.ndarray,
    membership: np.ndarray,
    sketch_nodes: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(baseline, dtype=np.uint64)
    membership = np.asarray(membership, dtype=np.uint64)
    if baseline.shape[0] != membership.shape[0]:
        return {"sketch_status": "membership_length_mismatch"}
    baseline_best, membership_best = _best_partner_maps(baseline, membership)
    changed_nodes: list[int] = []
    for node, (baseline_cluster, membership_cluster) in enumerate(
        zip(baseline, membership, strict=False)
    ):
        baseline_cluster_int = int(baseline_cluster)
        membership_cluster_int = int(membership_cluster)
        baseline_aligned = baseline_best.get(baseline_cluster_int) == membership_cluster_int
        membership_aligned = membership_best.get(membership_cluster_int) == baseline_cluster_int
        if not (baseline_aligned and membership_aligned):
            changed_nodes.append(node)
    changed_support = stable_sample_nodes(
        np.asarray(changed_nodes, dtype=np.uint32),
        BASIN_CHANGED_SUPPORT_SKETCH_SAMPLE_SIZE,
    )
    alignment_error_node_count = int(len(changed_nodes))
    alignment_error_fraction = (
        float(alignment_error_node_count) / float(baseline.shape[0])
        if baseline.shape[0]
        else math.nan
    )
    return {
        "p5_changed_nodes_vs_baseline": alignment_error_node_count,
        "p5_changed_fraction_vs_baseline": alignment_error_fraction,
        ALIGNMENT_ERROR_NODE_COUNT_COLUMN: alignment_error_node_count,
        ALIGNMENT_ERROR_FRACTION_COLUMN: alignment_error_fraction,
        "p5_basin_sketch_sample_size": int(np.asarray(sketch_nodes).shape[0]),
        SKETCH_HASH_COLUMN: hash_u32_sequence(sketch_nodes),
        SKETCH_BASELINE_COLUMN: encode_membership_sketch(baseline, sketch_nodes),
        SKETCH_MEMBERSHIP_COLUMN: encode_membership_sketch(membership, sketch_nodes),
        "p5_basin_changed_support_node_count": alignment_error_node_count,
        ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN: alignment_error_node_count,
        "p5_basin_changed_support_sketch_sample_size": int(changed_support.shape[0]),
        "p5_basin_changed_support_node_hash": hash_u32_sequence(changed_support),
        CHANGED_SUPPORT_COLUMN: encode_u32_sequence(changed_support),
        ALIGNED_CHANGED_SUPPORT_NODES_COLUMN: encode_u32_sequence(changed_support),
        "sketch_status": "ok",
    }


def _run_id(case: str, seed: int, randomness: float, requested_n_iterations: str) -> str:
    return f"{case}|vanilla|seed={seed}|randomness={randomness:g}|n={requested_n_iterations}"


def _target_case_filter(target_rows_path: Path | None, target_classes: set[str]) -> set[str]:
    if target_rows_path is None or not target_rows_path.exists():
        return set()
    targets = pd.read_csv(target_rows_path)
    if target_classes and "target_class" in targets.columns:
        targets = targets[targets["target_class"].astype(str).isin(target_classes)]
    if "case" not in targets.columns:
        return set()
    return {
        str(value)
        for value in targets["case"].dropna().astype(str).tolist()
        if value
    }


def _manifest_rows(
    case_manifest: Path,
    *,
    target_cases: set[str],
    fields: set[int],
    methods: set[str],
    max_cases: int | None,
) -> list[dict[str, Any]]:
    manifest = pd.read_csv(case_manifest)
    rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        graph_dir = Path(str(item.get("graph_dir", ""))).expanduser()
        if not graph_dir.is_absolute():
            graph_dir = (REPO_ROOT / graph_dir).resolve()
        case = _case_name(graph_dir)
        field = _safe_int(item.get("field"))
        method = str(item.get("method", ""))
        if target_cases and case not in target_cases:
            continue
        if fields and field not in fields:
            continue
        if methods and method not in methods:
            continue
        if str(item.get("status", "completed")) not in {"completed", "nan"}:
            continue
        rows.append(
            {
                "case": case,
                "field": field,
                "method": method,
                "sample": graph_dir.parent.name,
                "graph_dir": graph_dir,
                "case_slug": item.get("case_slug", ""),
            }
        )
        if max_cases is not None and len(rows) >= max_cases:
            break
    return rows


def _existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path, usecols=["run_id"])
    except (pd.errors.EmptyDataError, ValueError):
        return set()
    return set(frame["run_id"].dropna().astype(str).tolist())


def _load_graph(graph_dir: Path) -> tuple[Any, np.ndarray, Any]:
    arrays = _load_graph_arrays(graph_dir)
    node_weights = np.asarray(arrays.node_weights, dtype=np.float64)
    graph = build_leiden_graph(
        edges_src=arrays.src,
        edges_dst=arrays.dst,
        edges_weight=arrays.weight,
        n_nodes=int(node_weights.shape[0]),
        node_weights=node_weights,
    )
    return graph, node_weights, arrays


def _read_candidate_rows(candidate_dirs: tuple[Path, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for directory in candidate_dirs:
        root = directory.expanduser().resolve()
        for path in sorted(root.glob("**/candidate_level_rows.csv")):
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if frame.empty:
                continue
            frame["candidate_source_path"] = str(path.relative_to(root))
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "candidate_index" in out.columns:
        out["candidate_index"] = pd.to_numeric(out["candidate_index"], errors="coerce")
    return out


def _case_candidate_rows(candidates: pd.DataFrame, case: str) -> pd.DataFrame:
    if candidates.empty or "case" not in candidates.columns:
        return pd.DataFrame()
    out = candidates[candidates["case"].astype(str) == str(case)].copy()
    if out.empty:
        return out
    if "candidate_index" in out.columns:
        out = out.sort_values("candidate_index", na_position="last")
    return out


def _neighbor_clusters_for_nodes(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    membership: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    if nodes.size == 0:
        return np.asarray([], dtype=np.uint64)
    mask = np.zeros(int(membership.shape[0]), dtype=bool)
    mask[np.asarray(nodes, dtype=np.int64)] = True
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    src_hit = mask[src_arr]
    dst_hit = mask[dst_arr]
    neighbors = np.concatenate([dst_arr[src_hit], src_arr[dst_hit]])
    if neighbors.size == 0:
        return np.asarray([], dtype=np.uint64)
    return np.asarray(membership[neighbors], dtype=np.uint64)


def compatible_sketch_nodes(
    *,
    arrays: Any,
    baseline_membership: np.ndarray,
    node_weights: np.ndarray,
    candidate_rows: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate_rows.empty:
        return np.asarray([], dtype=np.uint32), {"sketch_context_status": "missing_candidate_rows"}
    baseline_membership = np.asarray(baseline_membership, dtype=np.uint64)
    active_clusters: set[int] = set()
    reconstructed_count = 0
    for _, row in candidate_rows.iterrows():
        source = _safe_int(row.get("source_cluster"))
        target = _safe_int(row.get("target_cluster"))
        if source is None or target is None:
            continue
        active_clusters.add(int(source))
        active_clusters.add(int(target))
        group_nodes, reconstruction = _reconstruct_external_group(
            src=arrays.src,
            dst=arrays.dst,
            weight=arrays.weight,
            membership=baseline_membership,
            node_weights=node_weights,
            source_cluster=int(source),
            target_cluster=int(target),
        )
        if str(reconstruction.get("reconstruction_status")) != "ok":
            continue
        reconstructed_count += 1
        if group_nodes.size == 0:
            continue
        active_clusters.update(int(value) for value in baseline_membership[group_nodes])
        neighbor_clusters = _neighbor_clusters_for_nodes(
            src=arrays.src,
            dst=arrays.dst,
            membership=baseline_membership,
            nodes=group_nodes,
        )
        active_clusters.update(int(value) for value in neighbor_clusters)
    if not active_clusters:
        nodes = np.arange(int(baseline_membership.shape[0]), dtype=np.uint32)
    else:
        active = np.asarray(sorted(active_clusters), dtype=np.uint64)
        nodes = np.flatnonzero(np.isin(baseline_membership, active)).astype(np.uint32)
    sketch_nodes = stable_sample_nodes(nodes, BASIN_SIGNATURE_SKETCH_SAMPLE_SIZE)
    candidate_hashes: list[str] = []
    if SKETCH_HASH_COLUMN in candidate_rows.columns:
        candidate_hashes = sorted(
            {
                str(value)
                for value in candidate_rows[SKETCH_HASH_COLUMN].dropna().tolist()
                if str(value)
            }
        )
    sketch_hash = hash_u32_sequence(sketch_nodes)
    return sketch_nodes, {
        "sketch_context_status": "ok" if sketch_nodes.size else "empty_sketch",
        "sketch_context_candidate_rows": int(len(candidate_rows)),
        "sketch_context_reconstructed_candidates": int(reconstructed_count),
        "sketch_context_active_clusters": int(len(active_clusters)),
        "sketch_context_candidate_hash_count": int(len(candidate_hashes)),
        "sketch_context_hash_matches_candidate": (
            bool(sketch_hash in candidate_hashes) if candidate_hashes else False
        ),
    }


def _run_one(
    *,
    graph: Any,
    case_row: dict[str, Any],
    seed: int,
    randomness: float,
    budget: IterationBudget,
    resolution: float,
    baseline_membership: np.ndarray | None = None,
    sketch_nodes: np.ndarray | None = None,
    sketch_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    result = graph.run_leiden(
        resolution=float(resolution),
        seed=int(seed),
        n_iterations=int(budget.n_iterations),
        randomness=float(randomness),
    )
    elapsed_sec = time.perf_counter() - start
    membership = np.asarray(result.membership, dtype=np.uint64)
    signature, cluster_count = canonical_partition_signature(membership)
    row = {
        "case": case_row["case"],
        "field": case_row["field"],
        "method": case_row["method"],
        "sample": case_row["sample"],
        "graph_dir": str(case_row["graph_dir"]),
        "variant": VARIANT_STANDARD,
        "run_id": _run_id(case_row["case"], seed, randomness, budget.requested),
        "seed": int(seed),
        "randomness": float(randomness),
        "requested_n_iterations": budget.requested,
        "iteration_mode": budget.mode,
        "n_iterations": int(budget.n_iterations),
        "resolution": float(resolution),
        "elapsed_sec": float(elapsed_sec),
        "quality": float(result.quality),
        "n_clusters": int(result.n_clusters),
        "p5_basin_signature": signature,
        "p5_basin_cluster_count": int(cluster_count),
        "comparison_scope": (
            "exact_signature_plus_compatible_sketch"
            if baseline_membership is not None and sketch_nodes is not None
            else "exact_signature_only"
        ),
    }
    if sketch_context:
        row.update(sketch_context)
    if baseline_membership is not None and sketch_nodes is not None and sketch_nodes.size:
        row.update(
            basin_sketch_stats(
                baseline=baseline_membership,
                membership=membership,
                sketch_nodes=sketch_nodes,
            )
        )
    elif baseline_membership is not None:
        row["sketch_status"] = "missing_sketch_nodes"
    return row


def collect_sweep(
    *,
    case_manifest: Path,
    target_rows_path: Path | None,
    output_dir: Path,
    seeds: tuple[int, ...],
    randomness_values: tuple[float, ...],
    n_iterations_values: tuple[IterationBudget, ...],
    target_classes: set[str],
    fields: set[int],
    methods: set[str],
    candidate_dirs: tuple[Path, ...] = (),
    max_cases: int | None = None,
    run_limit: int | None = None,
    resolution: float = 0.01,
    compatible_sketches: bool = False,
    sketch_baseline_seed: int = 11,
    sketch_baseline_iterations: int = 10,
    sketch_baseline_randomness: float = 0.01,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / ROWS_FILENAME
    target_cases = _target_case_filter(target_rows_path, target_classes)
    cases = _manifest_rows(
        case_manifest,
        target_cases=target_cases,
        fields=fields,
        methods=methods,
        max_cases=max_cases,
    )
    existing_run_ids = _existing_run_ids(rows_path) if resume else set()
    candidate_rows = _read_candidate_rows(candidate_dirs) if compatible_sketches else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if resume and rows_path.exists():
        try:
            rows.extend(pd.read_csv(rows_path).to_dict("records"))
        except pd.errors.EmptyDataError:
            pass

    run_count = 0
    skipped_count = 0
    for case_row in cases:
        graph, node_weights, arrays = _load_graph(Path(case_row["graph_dir"]))
        baseline_membership = None
        sketch_nodes = None
        sketch_context: dict[str, Any] | None = None
        if compatible_sketches:
            case_candidates = _case_candidate_rows(candidate_rows, str(case_row["case"]))
            baseline = graph.run_leiden(
                resolution=float(resolution),
                seed=int(sketch_baseline_seed),
                n_iterations=int(sketch_baseline_iterations),
                randomness=float(sketch_baseline_randomness),
                membership_dtype=np.uint32,
            )
            baseline_membership = np.asarray(baseline.membership, dtype=np.uint64)
            sketch_nodes, sketch_context = compatible_sketch_nodes(
                arrays=arrays,
                baseline_membership=baseline_membership,
                node_weights=node_weights,
                candidate_rows=case_candidates,
            )
        for seed in seeds:
            for randomness in randomness_values:
                for budget in n_iterations_values:
                    run_id = _run_id(case_row["case"], seed, randomness, budget.requested)
                    if run_id in existing_run_ids:
                        skipped_count += 1
                        continue
                    if run_limit is not None and run_count >= run_limit:
                        break
                    row = _run_one(
                        graph=graph,
                        case_row=case_row,
                        seed=int(seed),
                        randomness=float(randomness),
                        budget=budget,
                        resolution=float(resolution),
                        baseline_membership=baseline_membership,
                        sketch_nodes=sketch_nodes,
                        sketch_context=sketch_context,
                    )
                    rows.append(row)
                    _write_csv(rows_path, rows)
                    run_count += 1
                if run_limit is not None and run_count >= run_limit:
                    break
            if run_limit is not None and run_count >= run_limit:
                break
        if run_limit is not None and run_count >= run_limit:
            break

    _write_csv(rows_path, rows)
    summary = {
        "schema": "leiden_vanilla_reachability_sweep.v1",
        "case_manifest": str(case_manifest),
        "target_rows": "" if target_rows_path is None else str(target_rows_path),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "output_dir": str(output_dir),
        "target_classes": sorted(target_classes),
        "fields": sorted(fields),
        "methods": sorted(methods),
        "seeds": list(seeds),
        "randomness_values": list(randomness_values),
        "n_iterations_values": [
            {
                "requested_n_iterations": budget.requested,
                "iteration_mode": budget.mode,
                "n_iterations": budget.n_iterations,
            }
            for budget in n_iterations_values
        ],
        "candidate_case_count": len(cases),
        "run_count": run_count,
        "skipped_existing_count": skipped_count,
        "row_count": len(rows),
        "comparison_scope": (
            "exact_signature_plus_compatible_sketch"
            if compatible_sketches
            else "exact_signature_only"
        ),
        "compatible_sketches": bool(compatible_sketches),
        "sketch_baseline": {
            "seed": int(sketch_baseline_seed),
            "n_iterations": int(sketch_baseline_iterations),
            "randomness": float(sketch_baseline_randomness),
        },
        "paths": {
            "rows": str(rows_path),
            "summary": str(output_dir / SUMMARY_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_FILENAME, summary, rows)
    return summary


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    lines = [
        "# Vanilla Leiden Basin Reachability Sweep",
        "",
        "This artifact supports Dongdaemun basin reachability audits by collecting same-case standard Leiden partition signatures.",
        "",
        "## Scope",
        "",
        f"- Candidate cases: {summary['candidate_case_count']}",
        f"- Rows: {summary['row_count']}",
        f"- New runs: {summary['run_count']}",
        f"- Skipped existing runs: {summary['skipped_existing_count']}",
        f"- Comparison scope: {summary['comparison_scope']}",
        f"- Compatible sketches: {summary['compatible_sketches']}",
        "",
        "## Caution",
        "",
        "- `exact_signature_only` can prove exact partition reachability.",
        "- A non-match does not rule out coarse/near basin reachability because candidate-local sketch nodes are not reconstructed here.",
    ]
    if summary["compatible_sketches"]:
        lines[-1] = "- Compatible sketches can evaluate coarse/near basin reachability when `sketch_status=ok` and the sketch hash matches target rows."
    if not frame.empty:
        grouped = (
            frame.groupby(["field", "method"], dropna=False)
            .agg(rows=("run_id", "count"), distinct_signatures=("p5_basin_signature", "nunique"))
            .reset_index()
        )
        lines.extend(["", "## Field/Method Summary", ""])
        lines.extend(_markdown_table(grouped))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_CASE_MANIFEST)
    parser.add_argument("--target-rows", type=Path, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-dir",
        dest="candidate_dirs",
        action="append",
        type=Path,
        default=[],
        help="Candidate artifact directory containing candidate_level_rows.csv. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--target-classes",
        default="material_winner,core_alternative",
        help="Comma-separated target classes used to restrict manifest cases.",
    )
    parser.add_argument("--fields", default="", help="Optional comma-separated fields.")
    parser.add_argument("--methods", default="", help="Optional comma-separated methods.")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated Leiden seeds.",
    )
    parser.add_argument(
        "--randomness-values",
        default=",".join(str(value) for value in DEFAULT_RANDOMNESS),
        help="Comma-separated refinement randomness values.",
    )
    parser.add_argument(
        "--n-iterations-values",
        default=",".join(DEFAULT_N_ITERATIONS_VALUES),
        help="Comma-separated iteration budgets; use convergence for n_iterations=0.",
    )
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--compatible-sketches", action="store_true")
    parser.add_argument("--sketch-baseline-seed", type=int, default=11)
    parser.add_argument("--sketch-baseline-iterations", type=int, default=10)
    parser.add_argument("--sketch-baseline-randomness", type=float, default=0.01)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--run-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = collect_sweep(
        case_manifest=args.case_manifest,
        target_rows_path=args.target_rows,
        output_dir=args.output_dir,
        candidate_dirs=tuple(args.candidate_dirs),
        seeds=tuple(_parse_csv(args.seeds, cast=int)),
        randomness_values=tuple(_parse_csv(args.randomness_values, cast=float)),
        n_iterations_values=_parse_n_iterations_values(args.n_iterations_values),
        target_classes=set(_parse_csv(args.target_classes)),
        fields=set(_parse_csv(args.fields, cast=int)),
        methods=set(_parse_csv(args.methods)),
        max_cases=args.max_cases,
        run_limit=args.run_limit,
        resolution=float(args.resolution),
        compatible_sketches=bool(args.compatible_sketches),
        sketch_baseline_seed=int(args.sketch_baseline_seed),
        sketch_baseline_iterations=int(args.sketch_baseline_iterations),
        sketch_baseline_randomness=float(args.sketch_baseline_randomness),
        resume=bool(args.resume),
    )
    print(json.dumps(summary["paths"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
