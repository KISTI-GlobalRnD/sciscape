"""Probe forced high-gamma splits followed by baseline-gamma merge repair."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import sciscape_leiden
from sciscape.clustering import hierarchy_postprocess as _hierarchy_postprocess
from sciscape.clustering.adaptive_refinement import (
    SplitRepairSelectionPolicy,
    rank_split_repair_candidates,
    write_split_repair_candidate_selection_report,
)


FIELDS = [
    "rank",
    "cluster",
    "gamma_multiplier",
    "probe_resolution",
    "block_count",
    "doc_weight",
    "induced_directed_edges",
    "n_parts",
    "core_part_count",
    "singleton_weight",
    "cut_weight",
    "split_delta_q_base",
    "split_delta_q_probe",
    "repair_quotient_edges",
    "repair_merge_count",
    "repair_delta_q",
    "net_delta_q",
    "final_source_units",
    "retained_source_units",
    "escaped_source_units",
    "escaped_source_weight",
    "final_small_source_units",
    "final_small_source_weight",
    "largest_source_unit_fraction",
    "restored_source_cluster",
]

APPLY_FIELDS = [
    "selection_rank",
    "selected_index",
    "cluster",
    "gamma_multiplier",
    "probe_resolution",
    "block_count",
    "doc_weight",
    "n_parts",
    "split_delta_q_base",
    "repair_delta_q",
    "predicted_net_delta_q",
    "repair_merge_count",
    "final_source_units",
    "retained_source_units",
    "escaped_source_units",
    "escaped_source_weight",
    "final_small_source_units",
    "final_small_source_weight",
    "largest_source_unit_fraction",
    "changed_nodes",
    "moved_to_existing_cluster_nodes",
    "moved_to_new_cluster_nodes",
    "new_retained_clusters",
]

TRIM_FIELDS = [
    "move_index",
    "committed",
    "source",
    "target",
    "node",
    "node_weight",
    "delta_q",
    "source_weight_before",
    "source_weight_after",
    "target_weight_before",
    "target_weight_after",
]

OVERSIZE_ACCEPTANCE_MODES = ("quality_first", "hard_cap")
HARD_CAP_DEFAULT_TRIM_MIN_DELTA_Q = -1.0


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return 0.0


def _hwm_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _phase(name: str, phases: list[dict], fn):
    _log(f"phase_start {name} rss={_rss_mb():.1f}MB hwm={_hwm_mb():.1f}MB")
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    entry = {
        "name": name,
        "elapsed_sec": elapsed,
        "rss_mb": _rss_mb(),
        "hwm_mb": _hwm_mb(),
    }
    phases.append(entry)
    _log(
        f"phase_done {name} elapsed={elapsed:.2f}s "
        f"rss={entry['rss_mb']:.1f}MB hwm={entry['hwm_mb']:.1f}MB"
    )
    return result


def _load_membership(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["node_idx", "cluster"])
    node_idx = table.column("node_idx").combine_chunks().to_numpy(zero_copy_only=False)
    cluster = table.column("cluster").combine_chunks().to_numpy(zero_copy_only=False)
    order = np.argsort(node_idx)
    return np.asarray(cluster[order], dtype=np.uint64)


def _load_node_weights(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float64)


def _load_candidates(path: Path, policy: str | None, max_candidates: int) -> np.ndarray:
    seen: set[int] = set()
    clusters: list[int] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if policy and row.get("policy") != policy:
                continue
            cluster = int(row["cluster"])
            if cluster in seen:
                continue
            seen.add(cluster)
            clusters.append(cluster)
            if len(clusters) >= max_candidates:
                break
    return np.asarray(clusters, dtype=np.uint64)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _cluster_weight_arrays(
    membership: np.ndarray,
    node_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    membership_i64 = np.asarray(membership, dtype=np.int64)
    minlength = int(membership_i64.max()) + 1 if membership_i64.size else 0
    counts = np.bincount(membership_i64, minlength=minlength)
    weights = np.bincount(membership_i64, weights=node_weights, minlength=minlength)
    return counts, weights


def _membership_weight_summary(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    min_weight: float = 0.0,
    max_weight: float,
) -> dict:
    counts, weights = _cluster_weight_arrays(membership, node_weights)
    active = counts > 0
    active_counts = counts[active]
    active_weights = weights[active]
    return {
        "n_clusters": int(active_weights.size),
        "max_doc_weight": float(active_weights.max()) if active_weights.size else 0.0,
        "n_above_max_doc_weight": (
            int((active_weights > max_weight).sum()) if max_weight > 0.0 else 0
        ),
        "n_singletons": int((active_counts == 1).sum()),
        "n_lt_min_doc_weight": (
            int((active_weights < min_weight).sum()) if min_weight > 0.0 else 0
        ),
        "n_lt_25_doc_weight": int((active_weights < 25.0).sum()),
        "n_lt_50_doc_weight": int((active_weights < 50.0).sum()),
        "top10_doc_weights": [float(x) for x in np.sort(active_weights)[::-1][:10]],
    }


def _postprocess_policy_summary(args: argparse.Namespace) -> dict:
    return {
        "acceptance_mode": str(args.oversize_acceptance_mode),
        "target_min_doc_weight": float(args.target_min_doc_weight),
        "target_max_doc_weight": float(args.target_max_doc_weight),
        "selection_mode": str(args.selection_mode),
        "selection_singleton_budget": float(args.selection_singleton_budget),
        "selection_max_selected": int(args.selection_max_selected),
        "apply_split_repair_candidates": bool(args.apply_split_repair_candidates),
        "apply_iterations": int(args.apply_iterations),
        "apply_min_quality_delta": float(args.apply_min_quality_delta),
        "apply_oversize_boundary_trim": bool(args.apply_oversize_boundary_trim),
        "trim_min_delta_q": float(args.trim_min_delta_q),
        "trim_min_delta_q_source": str(
            getattr(args, "trim_min_delta_q_source", "explicit")
        ),
        "trim_max_moves_per_cluster": int(args.trim_max_moves_per_cluster),
        "pair_seeded_probes": bool(args.pair_seeded_probes),
    }


def _small_membership_view(stats: dict) -> dict:
    return {
        "n_clusters": int(stats["n_clusters"]),
        "n_singletons": int(stats["n_singletons"]),
        "n_lt_min_doc_weight": int(stats["n_lt_min_doc_weight"]),
    }


def _oversize_membership_view(stats: dict) -> dict:
    return {
        "n_clusters": int(stats["n_clusters"]),
        "n_above_max_doc_weight": int(stats["n_above_max_doc_weight"]),
        "max_doc_weight": float(stats["max_doc_weight"]),
        "top10_doc_weights": [float(x) for x in stats.get("top10_doc_weights", [])],
    }


def _membership_delta(after: dict, before: dict, keys: list[str]) -> dict:
    return {key: after[key] - before[key] for key in keys}


def _oversize_residual_summary(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    max_weight: float,
    top_k: int = 10,
) -> dict:
    if max_weight <= 0.0:
        return {
            "n_above_max_doc_weight": 0,
            "max_doc_weight": 0.0,
            "excess_doc_weight_total": 0.0,
            "top_excess_clusters": [],
        }
    counts, weights = _cluster_weight_arrays(membership, node_weights)
    active = (counts > 0) & (weights > max_weight)
    clusters = np.flatnonzero(active)
    if clusters.size == 0:
        return {
            "n_above_max_doc_weight": 0,
            "max_doc_weight": float(weights[counts > 0].max()) if np.any(counts > 0) else 0.0,
            "excess_doc_weight_total": 0.0,
            "top_excess_clusters": [],
        }
    excess = weights[clusters] - max_weight
    order = np.lexsort((clusters, -excess))
    top_clusters = clusters[order[:top_k]]
    return {
        "n_above_max_doc_weight": int(clusters.size),
        "max_doc_weight": float(weights[clusters].max()),
        "excess_doc_weight_total": float(excess.sum()),
        "top_excess_clusters": [
            {
                "cluster": int(cluster),
                "doc_weight": float(weights[cluster]),
                "excess_doc_weight": float(weights[cluster] - max_weight),
            }
            for cluster in top_clusters
        ],
    }


def _trim_source_move_counts(
    raw_trim: dict[str, np.ndarray],
    candidate_clusters: np.ndarray,
    n_moves: int,
) -> list[dict]:
    counts = {int(cluster): 0 for cluster in np.asarray(candidate_clusters).tolist()}
    if n_moves > 0:
        sources = np.asarray(raw_trim["source"][:n_moves], dtype=np.uint64)
        unique, move_counts = np.unique(sources, return_counts=True)
        for source, count in zip(unique, move_counts, strict=False):
            counts[int(source)] = int(count)
    return [
        {"cluster": cluster, "moves": moves}
        for cluster, moves in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _trim_infeasibility_diagnostics(
    *,
    raw_trim: dict[str, np.ndarray],
    candidate_clusters: np.ndarray,
    committed_membership: np.ndarray,
    proposed_membership: np.ndarray,
    node_weights: np.ndarray,
    target_max_weight: float,
    trim_min_delta_q: float,
    max_moves_per_cluster: int,
    n_moves_committed: int,
    n_moves_proposed: int,
    quality_floor: float,
    quality_after_committed: float,
    quality_after_proposed: float,
) -> dict:
    committed_residual = _oversize_residual_summary(
        committed_membership,
        node_weights,
        max_weight=target_max_weight,
    )
    proposed_residual = _oversize_residual_summary(
        proposed_membership,
        node_weights,
        max_weight=target_max_weight,
    )
    target_max_satisfied = committed_residual["n_above_max_doc_weight"] == 0
    proposed_target_max_satisfied = proposed_residual["n_above_max_doc_weight"] == 0
    quality_floor_limited = bool(n_moves_committed < n_moves_proposed)
    source_moves_proposed = _trim_source_move_counts(
        raw_trim,
        candidate_clusters,
        n_moves_proposed,
    )
    move_budget_exhausted = (
        max_moves_per_cluster > 0
        and any(row["moves"] >= max_moves_per_cluster for row in source_moves_proposed)
        and not proposed_target_max_satisfied
    )

    inferred_blockers: list[str] = []
    if target_max_satisfied:
        inferred_blockers.append("target_satisfied")
    else:
        if n_moves_proposed == 0:
            inferred_blockers.append("no_candidate_boundary_moves")
        if quality_floor_limited:
            inferred_blockers.append("quality_floor")
        if move_budget_exhausted:
            inferred_blockers.append("move_budget")
        if (
            n_moves_proposed > 0
            and not proposed_target_max_satisfied
            and not move_budget_exhausted
        ):
            inferred_blockers.append("trim_delta_bound_or_receiver_cap")
        if not inferred_blockers:
            inferred_blockers.append("unclassified")

    return {
        "target_max_satisfied": bool(target_max_satisfied),
        "proposed_target_max_satisfied": bool(proposed_target_max_satisfied),
        "quality_floor_limited": quality_floor_limited,
        "quality_floor_margin_committed": float(quality_after_committed - quality_floor),
        "quality_floor_margin_proposed": float(quality_after_proposed - quality_floor),
        "move_budget_exhausted": bool(move_budget_exhausted),
        "trim_min_delta_q": float(trim_min_delta_q),
        "max_moves_per_cluster": int(max_moves_per_cluster),
        "n_moves_committed": int(n_moves_committed),
        "n_moves_proposed": int(n_moves_proposed),
        "source_move_counts_committed": _trim_source_move_counts(
            raw_trim,
            candidate_clusters,
            n_moves_committed,
        ),
        "source_move_counts_proposed": source_moves_proposed,
        "committed_oversize_residual": committed_residual,
        "proposed_oversize_residual": proposed_residual,
        "inferred_blockers": inferred_blockers,
    }


def _postprocess_transition_report(
    args: argparse.Namespace,
    before_stats: dict,
    after_stats: dict,
    *,
    changed_nodes: int,
    split_repair_exact_delta_q: float,
    trim_exact_delta_q: float,
    final_exact_delta_q: float,
    stop_reason: str,
) -> dict:
    small_before = _small_membership_view(before_stats)
    small_after = _small_membership_view(after_stats)
    oversize_before = _oversize_membership_view(before_stats)
    oversize_after = _oversize_membership_view(after_stats)
    target_max_satisfied = (
        float(args.target_max_doc_weight) <= 0.0
        or oversize_after["n_above_max_doc_weight"] == 0
    )
    return {
        "acceptance_mode": str(args.oversize_acceptance_mode),
        "postprocess_policy": _postprocess_policy_summary(args),
        "target_max_satisfied": bool(target_max_satisfied),
        "small_cluster_summary": {
            "before": small_before,
            "after": small_after,
            "delta": _membership_delta(
                small_after,
                small_before,
                ["n_clusters", "n_singletons", "n_lt_min_doc_weight"],
            ),
            "changed_nodes": int(changed_nodes),
            "stop_reason": stop_reason,
        },
        "oversize_summary": {
            "before": oversize_before,
            "after": oversize_after,
            "delta": {
                "n_clusters": (
                    oversize_after["n_clusters"] - oversize_before["n_clusters"]
                ),
                "n_above_max_doc_weight": (
                    oversize_after["n_above_max_doc_weight"]
                    - oversize_before["n_above_max_doc_weight"]
                ),
                "max_doc_weight": (
                    oversize_after["max_doc_weight"]
                    - oversize_before["max_doc_weight"]
                ),
            },
            "changed_nodes": int(changed_nodes),
            "split_repair_exact_delta_q": float(split_repair_exact_delta_q),
            "trim_exact_delta_q": float(trim_exact_delta_q),
            "final_exact_delta_q": float(final_exact_delta_q),
            "target_max_satisfied": bool(target_max_satisfied),
            "stop_reason": stop_reason,
        },
    }


def _current_oversize_candidate_clusters(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    max_weight: float,
    max_candidates: int,
) -> np.ndarray:
    if max_weight <= 0.0:
        return np.asarray([], dtype=np.uint64)
    counts, weights = _cluster_weight_arrays(membership, node_weights)
    candidate_clusters = np.flatnonzero((counts > 0) & (weights > max_weight))
    order = np.lexsort((candidate_clusters, -weights[candidate_clusters]))
    if max_candidates > 0:
        order = order[:max_candidates]
    return np.asarray(candidate_clusters[order], dtype=np.uint64)


def _summary(raw: dict[str, np.ndarray], phases: list[dict], paths: dict[str, str]) -> dict:
    net_positive = raw["net_delta_q"] > 0
    net_positive_eps = raw["net_delta_q"] > 1e-6
    net_positive_one = raw["net_delta_q"] > 1.0
    escaped = raw["escaped_source_units"] > 0
    restored = raw["restored_source_cluster"]
    return {
        "n_probes": int(raw["cluster"].shape[0]),
        "n_net_positive": int(net_positive.sum()),
        "n_net_positive_gt_1e_minus_6": int(net_positive_eps.sum()),
        "n_net_positive_gt_1": int(net_positive_one.sum()),
        "n_net_positive_escaped_gt_1e_minus_6": int((net_positive_eps & escaped).sum()),
        "n_with_repair_merges": int((raw["repair_merge_count"] > 0).sum()),
        "n_with_escaped_source": int(escaped.sum()),
        "n_restored_source_cluster": int(restored.sum()),
        "n_retained_split": int((raw["retained_source_units"] >= 2).sum()),
        "net_delta_q": {
            "p50": _percentile(raw["net_delta_q"], 50),
            "p90": _percentile(raw["net_delta_q"], 90),
            "p95": _percentile(raw["net_delta_q"], 95),
            "p99": _percentile(raw["net_delta_q"], 99),
            "max": float(raw["net_delta_q"].max()) if raw["net_delta_q"].size else 0.0,
        },
        "repair_delta_q": {
            "p50": _percentile(raw["repair_delta_q"], 50),
            "p90": _percentile(raw["repair_delta_q"], 90),
            "p95": _percentile(raw["repair_delta_q"], 95),
            "p99": _percentile(raw["repair_delta_q"], 99),
        },
        "escaped_source_weight": {
            "p50": _percentile(raw["escaped_source_weight"], 50),
            "p90": _percentile(raw["escaped_source_weight"], 90),
            "p95": _percentile(raw["escaped_source_weight"], 95),
        },
        "largest_source_unit_fraction": {
            "p50": _percentile(raw["largest_source_unit_fraction"], 50),
            "p90": _percentile(raw["largest_source_unit_fraction"], 90),
            "p95": _percentile(raw["largest_source_unit_fraction"], 95),
        },
        "phases": phases,
        "paths": paths,
        "rss_mb_final": _rss_mb(),
        "hwm_mb_final": _hwm_mb(),
    }


def _selection_policy(args: argparse.Namespace) -> SplitRepairSelectionPolicy:
    return SplitRepairSelectionPolicy(
        name=str(args.selection_mode),
        mode=str(args.selection_mode),
        singleton_budget=float(args.selection_singleton_budget),
        max_selected=(
            None
            if args.selection_max_selected <= 0
            else int(args.selection_max_selected)
        ),
    )


def _selection_rows(
    raw: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict]:
    return rank_split_repair_candidates(
        raw,
        _selection_policy(args),
        min_weight=float(args.target_min_doc_weight),
        max_weight=float(args.target_max_doc_weight),
    )


def _write_outputs(
    raw: dict[str, np.ndarray],
    output_dir: Path,
    phases: list[dict],
    args: argparse.Namespace,
    selection_rows: list[dict],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "split_merge_repair_probes.npz"
    np.savez(arrays_path, **raw)

    order = np.lexsort((raw["cluster"], -raw["net_delta_q"]))
    csv_path = output_dir / "split_merge_repair_probes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "cluster": int(raw["cluster"][idx]),
                    "gamma_multiplier": float(raw["gamma_multiplier"][idx]),
                    "probe_resolution": float(raw["probe_resolution"][idx]),
                    "block_count": int(raw["block_count"][idx]),
                    "doc_weight": float(raw["doc_weight"][idx]),
                    "induced_directed_edges": int(raw["induced_directed_edges"][idx]),
                    "n_parts": int(raw["n_parts"][idx]),
                    "core_part_count": int(raw["core_part_count"][idx]),
                    "singleton_weight": float(raw["singleton_weight"][idx]),
                    "cut_weight": float(raw["cut_weight"][idx]),
                    "split_delta_q_base": float(raw["split_delta_q_base"][idx]),
                    "split_delta_q_probe": float(raw["split_delta_q_probe"][idx]),
                    "repair_quotient_edges": int(raw["repair_quotient_edges"][idx]),
                    "repair_merge_count": int(raw["repair_merge_count"][idx]),
                    "repair_delta_q": float(raw["repair_delta_q"][idx]),
                    "net_delta_q": float(raw["net_delta_q"][idx]),
                    "final_source_units": int(raw["final_source_units"][idx]),
                    "retained_source_units": int(raw["retained_source_units"][idx]),
                    "escaped_source_units": int(raw["escaped_source_units"][idx]),
                    "escaped_source_weight": float(raw["escaped_source_weight"][idx]),
                    "final_small_source_units": int(raw["final_small_source_units"][idx]),
                    "final_small_source_weight": float(raw["final_small_source_weight"][idx]),
                    "largest_source_unit_fraction": float(
                        raw["largest_source_unit_fraction"][idx]
                    ),
                    "restored_source_cluster": bool(raw["restored_source_cluster"][idx]),
                }
            )
    selection_paths = write_split_repair_candidate_selection_report(
        raw,
        output_dir,
        _selection_policy(args),
        min_weight=float(args.target_min_doc_weight),
        max_weight=float(args.target_max_doc_weight),
        rows=selection_rows,
    )
    paths = {
        "arrays": str(arrays_path),
        "probes": str(csv_path),
        "selection_summary": selection_paths["summary"],
        "selection_candidates": selection_paths["candidates"],
    }
    summary = _summary(raw, phases, paths)
    summary["acceptance_mode"] = str(args.oversize_acceptance_mode)
    summary["postprocess_policy"] = _postprocess_policy_summary(args)
    summary_path = output_dir / "split_merge_repair_probe_summary.json"
    paths["summary"] = str(summary_path)
    summary["paths"] = paths
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _run_probe_selection_pass(
    graph,
    membership: np.ndarray,
    candidate_clusters: np.ndarray,
    gamma_multipliers: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    phases: list[dict],
) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    probe_kwargs = {
        "membership": membership,
        "candidate_clusters": candidate_clusters,
        "resolution": args.resolution,
        "gamma_multipliers": gamma_multipliers,
        "min_core_weight": args.min_core_weight,
        "randomness": args.randomness,
        "repair_epsilon": args.repair_epsilon,
        "seed": args.seed,
    }
    if args.pair_seeded_probes:
        probe_kwargs["pair_seeded"] = True
    raw = _phase(
        "split_merge_repair_probes",
        phases,
        lambda: graph.split_merge_repair_probes(**probe_kwargs),
    )
    raw = {key: np.asarray(value) for key, value in raw.items()}
    selection_rows = _phase(
        "split_repair_candidate_selection",
        phases,
        lambda: _selection_rows(raw, args),
    )
    summary = _phase(
        "write_outputs",
        phases,
        lambda: _write_outputs(raw, output_dir, phases, args, selection_rows),
    )
    return raw, selection_rows, summary


def _write_membership(path: Path, membership: np.ndarray) -> None:
    table = pa.table(
        {
            "node_idx": np.arange(membership.shape[0], dtype=np.uint64),
            "cluster": np.asarray(membership, dtype=np.uint64),
        }
    )
    pq.write_table(table, path, compression="zstd")


def _write_apply_candidate_rows(
    path: Path,
    raw_apply: dict[str, np.ndarray],
    selected_rows: list[dict],
) -> None:
    order = np.argsort(raw_apply["selected_index"], kind="stable")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=APPLY_FIELDS)
        writer.writeheader()
        for idx in order:
            selected_index = int(raw_apply["selected_index"][idx])
            selection_rank = int(selected_rows[selected_index]["rank"])
            writer.writerow(
                {
                    "selection_rank": selection_rank,
                    "selected_index": selected_index,
                    "cluster": int(raw_apply["cluster"][idx]),
                    "gamma_multiplier": float(raw_apply["gamma_multiplier"][idx]),
                    "probe_resolution": float(raw_apply["probe_resolution"][idx]),
                    "block_count": int(raw_apply["block_count"][idx]),
                    "doc_weight": float(raw_apply["doc_weight"][idx]),
                    "n_parts": int(raw_apply["n_parts"][idx]),
                    "split_delta_q_base": float(raw_apply["split_delta_q_base"][idx]),
                    "repair_delta_q": float(raw_apply["repair_delta_q"][idx]),
                    "predicted_net_delta_q": float(raw_apply["predicted_net_delta_q"][idx]),
                    "repair_merge_count": int(raw_apply["repair_merge_count"][idx]),
                    "final_source_units": int(raw_apply["final_source_units"][idx]),
                    "retained_source_units": int(raw_apply["retained_source_units"][idx]),
                    "escaped_source_units": int(raw_apply["escaped_source_units"][idx]),
                    "escaped_source_weight": float(raw_apply["escaped_source_weight"][idx]),
                    "final_small_source_units": int(raw_apply["final_small_source_units"][idx]),
                    "final_small_source_weight": float(
                        raw_apply["final_small_source_weight"][idx]
                    ),
                    "largest_source_unit_fraction": float(
                        raw_apply["largest_source_unit_fraction"][idx]
                    ),
                    "changed_nodes": int(raw_apply["changed_nodes"][idx]),
                    "moved_to_existing_cluster_nodes": int(
                        raw_apply["moved_to_existing_cluster_nodes"][idx]
                    ),
                    "moved_to_new_cluster_nodes": int(
                        raw_apply["moved_to_new_cluster_nodes"][idx]
                    ),
                    "new_retained_clusters": int(raw_apply["new_retained_clusters"][idx]),
                }
            )


def _write_trim_move_rows(
    path: Path,
    raw_trim: dict[str, np.ndarray],
    *,
    n_moves_committed: int | None = None,
) -> None:
    n_moves = int(raw_trim["node"].shape[0])
    if n_moves_committed is None:
        n_moves_committed = n_moves
    n_moves_committed = max(0, min(int(n_moves_committed), n_moves))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRIM_FIELDS)
        writer.writeheader()
        for idx in range(n_moves):
            writer.writerow(
                {
                    "move_index": idx,
                    "committed": idx < n_moves_committed,
                    "source": int(raw_trim["source"][idx]),
                    "target": int(raw_trim["target"][idx]),
                    "node": int(raw_trim["node"][idx]),
                    "node_weight": float(raw_trim["node_weight"][idx]),
                    "delta_q": float(raw_trim["delta_q"][idx]),
                    "source_weight_before": float(raw_trim["source_weight_before"][idx]),
                    "source_weight_after": float(raw_trim["source_weight_after"][idx]),
                    "target_weight_before": float(raw_trim["target_weight_before"][idx]),
                    "target_weight_after": float(raw_trim["target_weight_after"][idx]),
                }
            )


def _quality_floor_prefix_move_count(
    delta_q: np.ndarray,
    *,
    quality_before: float,
    quality_floor: float,
) -> int:
    """Return the longest move prefix whose predicted final quality meets floor."""

    deltas = np.asarray(delta_q, dtype=np.float64)
    if deltas.size == 0:
        return 0
    min_delta_q = float(quality_floor) - float(quality_before)
    cumulative = np.cumsum(deltas)
    valid = np.flatnonzero(cumulative >= min_delta_q - 1e-9)
    return int(valid[-1] + 1) if valid.size else 0


def _trim_prefix_membership(
    membership: np.ndarray,
    raw_trim: dict[str, np.ndarray],
    n_moves: int,
) -> np.ndarray:
    proposed = np.asarray(membership, dtype=np.uint64).copy()
    if n_moves <= 0:
        return proposed
    nodes = np.asarray(raw_trim["node"][:n_moves], dtype=np.int64)
    targets = np.asarray(raw_trim["target"][:n_moves], dtype=np.uint64)
    proposed[nodes] = targets
    return proposed


def _apply_selected_candidates(
    graph,
    membership: np.ndarray,
    candidate_clusters: np.ndarray,
    gamma_multipliers: np.ndarray,
    selection_rows: list[dict],
    output_dir: Path,
    args: argparse.Namespace,
    membership_output_path: Path | None = None,
) -> tuple[dict, np.ndarray | None]:
    selected_rows = [row for row in selection_rows if row["selected_for_apply"]]
    summary_path = output_dir / "split_repair_apply_summary.json"
    candidates_path = output_dir / "split_repair_apply_candidates.csv"
    if membership_output_path is not None:
        membership_path = membership_output_path
    elif args.applied_membership_output is not None:
        membership_path = args.applied_membership_output
    else:
        membership_path = output_dir / "split_repair_applied_membership.parquet"
    paths = {
        "apply_summary": str(summary_path),
        "apply_candidates": str(candidates_path),
    }
    if not selected_rows:
        with candidates_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=APPLY_FIELDS).writeheader()
        summary = {
            "status": "no_selected_candidates",
            "n_selected": 0,
            "n_applied": 0,
            "paths": paths,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary, None

    selected_clusters = np.asarray([row["cluster"] for row in selected_rows], dtype=np.uint64)
    selected_gamma_multipliers = np.asarray(
        [row["gamma_multiplier"] for row in selected_rows],
        dtype=np.float64,
    )
    apply_fn = getattr(graph, "apply_split_merge_repair_candidates", None)
    if apply_fn is None:
        raise AttributeError(
            "installed sciscape_leiden module does not expose "
            "Graph.apply_split_merge_repair_candidates"
        )

    apply_kwargs = {
        "membership": membership,
        "candidate_clusters": candidate_clusters,
        "selected_clusters": selected_clusters,
        "selected_gamma_multipliers": selected_gamma_multipliers,
        "resolution": float(args.resolution),
        "gamma_multipliers": gamma_multipliers,
        "min_core_weight": float(args.min_core_weight),
        "randomness": float(args.randomness),
        "repair_epsilon": float(args.repair_epsilon),
        "seed": int(args.seed),
    }
    if args.pair_seeded_probes:
        apply_kwargs["pair_seeded"] = True
    raw_apply = apply_fn(**apply_kwargs)
    raw_apply = {key: np.asarray(value) for key, value in raw_apply.items()}
    _write_apply_candidate_rows(candidates_path, raw_apply, selected_rows)

    quality_before = float(graph.cpm_quality(membership=membership, resolution=args.resolution))
    proposed_membership = np.asarray(raw_apply["membership"], dtype=np.uint64)
    quality_after = float(
        graph.cpm_quality(membership=proposed_membership, resolution=args.resolution)
    )
    exact_delta_q = quality_after - quality_before
    n_selected = len(selected_rows)
    n_applied = int(raw_apply["cluster"].shape[0])
    predicted_delta_q_sum = (
        float(raw_apply["predicted_net_delta_q"].sum()) if n_applied else 0.0
    )
    missing_candidates = n_selected - n_applied
    status = "committed"
    if missing_candidates:
        status = "rolled_back_missing_candidates"
    elif exact_delta_q < float(args.apply_min_quality_delta):
        status = "rolled_back_quality_below_threshold"

    if status == "committed":
        membership_path.parent.mkdir(parents=True, exist_ok=True)
        _write_membership(membership_path, proposed_membership)
        paths["applied_membership"] = str(membership_path)

    summary = {
        "status": status,
        "n_selected": n_selected,
        "n_applied": n_applied,
        "n_missing_candidates": int(missing_candidates),
        "quality_before": quality_before,
        "quality_after_proposed": quality_after,
        "exact_delta_q": exact_delta_q,
        "min_quality_delta": float(args.apply_min_quality_delta),
        "predicted_delta_q_sum": predicted_delta_q_sum,
        "changed_nodes": (
            int(raw_apply["changed_nodes"].sum()) if n_applied else 0
        ),
        "moved_to_existing_cluster_nodes": (
            int(raw_apply["moved_to_existing_cluster_nodes"].sum()) if n_applied else 0
        ),
        "moved_to_new_cluster_nodes": (
            int(raw_apply["moved_to_new_cluster_nodes"].sum()) if n_applied else 0
        ),
        "new_retained_clusters": (
            int(raw_apply["new_retained_clusters"].sum()) if n_applied else 0
        ),
        "paths": paths,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary, proposed_membership if status == "committed" else None


def _apply_oversize_boundary_trim(
    graph,
    membership: np.ndarray,
    node_weights: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    quality_floor: float | None = None,
) -> tuple[dict, np.ndarray | None]:
    summary_path = output_dir / "oversize_boundary_trim_summary.json"
    moves_path = output_dir / "oversize_boundary_trim_moves.csv"
    membership_path = output_dir / "oversize_boundary_trim_membership.parquet"
    paths = {
        "trim_summary": str(summary_path),
        "trim_moves": str(moves_path),
    }
    candidate_clusters = _current_oversize_candidate_clusters(
        membership,
        node_weights,
        max_weight=float(args.target_max_doc_weight),
        max_candidates=int(args.max_candidates),
    )
    before_stats = _membership_weight_summary(
        membership,
        node_weights,
        min_weight=float(args.target_min_doc_weight),
        max_weight=float(args.target_max_doc_weight),
    )
    if candidate_clusters.size == 0:
        with moves_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=TRIM_FIELDS).writeheader()
        summary = {
            "status": "no_current_oversize_candidates",
            "acceptance_mode": str(args.oversize_acceptance_mode),
            "candidate_clusters": 0,
            "n_moves": 0,
            "n_moves_proposed": 0,
            "n_moves_committed": 0,
            "target_max_satisfied": before_stats["n_above_max_doc_weight"] == 0,
            "min_delta_q": float(args.trim_min_delta_q),
            "before_membership": before_stats,
            "after_membership": before_stats,
            "paths": paths,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary, None

    trim_fn = getattr(graph, "trim_oversize_boundary_moves", None)
    if trim_fn is None:
        raise AttributeError(
            "installed sciscape_leiden module does not expose "
            "Graph.trim_oversize_boundary_moves"
        )

    raw_trim = trim_fn(
        membership=membership,
        candidate_clusters=candidate_clusters,
        resolution=float(args.resolution),
        target_max_weight=float(args.target_max_doc_weight),
        min_delta_q=float(args.trim_min_delta_q),
        max_moves_per_cluster=int(args.trim_max_moves_per_cluster),
    )
    raw_trim = {key: np.asarray(value) for key, value in raw_trim.items()}

    quality_before = float(graph.cpm_quality(membership=membership, resolution=args.resolution))
    proposed_membership = np.asarray(raw_trim["membership"], dtype=np.uint64)
    quality_after_proposed = float(
        graph.cpm_quality(membership=proposed_membership, resolution=args.resolution)
    )
    exact_delta_q_proposed = quality_after_proposed - quality_before
    n_moves_proposed = int(raw_trim["node"].shape[0])
    predicted_delta_q_sum_proposed = (
        float(raw_trim["delta_q"].sum()) if n_moves_proposed else 0.0
    )
    effective_quality_floor = (
        float(quality_floor)
        if quality_floor is not None
        else quality_before + float(args.apply_min_quality_delta)
    )
    quality_floor_source = (
        "run_final" if quality_floor is not None else "trim_step"
    )
    after_stats = before_stats
    status = "no_trim_moves"
    trim_commit_reason = "none"
    n_moves_committed = 0
    committed_membership = np.asarray(membership, dtype=np.uint64)
    quality_after_committed = quality_before
    if n_moves_proposed and quality_after_proposed < effective_quality_floor:
        n_moves_committed = _quality_floor_prefix_move_count(
            raw_trim["delta_q"],
            quality_before=quality_before,
            quality_floor=effective_quality_floor,
        )
        committed_membership = _trim_prefix_membership(
            membership,
            raw_trim,
            n_moves_committed,
        )
        quality_after_committed = float(
            graph.cpm_quality(membership=committed_membership, resolution=args.resolution)
        )
        while n_moves_committed > 0 and quality_after_committed < effective_quality_floor:
            n_moves_committed -= 1
            committed_membership = _trim_prefix_membership(
                membership,
                raw_trim,
                n_moves_committed,
            )
            quality_after_committed = float(
                graph.cpm_quality(membership=committed_membership, resolution=args.resolution)
            )
        if n_moves_committed:
            status = "committed"
            trim_commit_reason = "quality_floor_prefix"
        else:
            status = "rolled_back_quality_below_threshold"
            trim_commit_reason = "quality_floor"
    elif n_moves_proposed:
        status = "committed"
        trim_commit_reason = "all_proposed"
        n_moves_committed = n_moves_proposed
        committed_membership = proposed_membership
        quality_after_committed = quality_after_proposed

    exact_delta_q = quality_after_committed - quality_before
    predicted_delta_q_sum = (
        float(raw_trim["delta_q"][:n_moves_committed].sum())
        if n_moves_committed
        else 0.0
    )
    _write_trim_move_rows(
        moves_path,
        raw_trim,
        n_moves_committed=n_moves_committed,
    )
    if status == "committed":
        membership_path.parent.mkdir(parents=True, exist_ok=True)
        _write_membership(membership_path, committed_membership)
        paths["trim_membership"] = str(membership_path)
        after_stats = _membership_weight_summary(
            committed_membership,
            node_weights,
            min_weight=float(args.target_min_doc_weight),
            max_weight=float(args.target_max_doc_weight),
        )
    proposed_stats = _membership_weight_summary(
        proposed_membership,
        node_weights,
        min_weight=float(args.target_min_doc_weight),
        max_weight=float(args.target_max_doc_weight),
    )
    trim_diagnostics = _trim_infeasibility_diagnostics(
        raw_trim=raw_trim,
        candidate_clusters=candidate_clusters,
        committed_membership=committed_membership,
        proposed_membership=proposed_membership,
        node_weights=node_weights,
        target_max_weight=float(args.target_max_doc_weight),
        trim_min_delta_q=float(args.trim_min_delta_q),
        max_moves_per_cluster=int(args.trim_max_moves_per_cluster),
        n_moves_committed=n_moves_committed,
        n_moves_proposed=n_moves_proposed,
        quality_floor=effective_quality_floor,
        quality_after_committed=quality_after_committed,
        quality_after_proposed=quality_after_proposed,
    )

    summary = {
        "status": status,
        "acceptance_mode": str(args.oversize_acceptance_mode),
        "candidate_clusters": int(candidate_clusters.size),
        "candidate_cluster_ids": [int(x) for x in candidate_clusters.tolist()],
        "n_moves": int(n_moves_committed),
        "n_moves_proposed": int(n_moves_proposed),
        "n_moves_committed": int(n_moves_committed),
        "quality_before": quality_before,
        "quality_after_proposed": quality_after_proposed,
        "quality_after_committed": quality_after_committed,
        "quality_floor": effective_quality_floor,
        "quality_floor_source": quality_floor_source,
        "target_max_satisfied": trim_diagnostics["target_max_satisfied"],
        "exact_delta_q": exact_delta_q,
        "exact_delta_q_proposed": exact_delta_q_proposed,
        "predicted_delta_q_sum": predicted_delta_q_sum,
        "predicted_delta_q_sum_proposed": predicted_delta_q_sum_proposed,
        "quality_floor_limited": bool(n_moves_committed < n_moves_proposed),
        "trim_commit_reason": trim_commit_reason,
        "min_delta_q": float(args.trim_min_delta_q),
        "max_moves_per_cluster": int(args.trim_max_moves_per_cluster),
        "changed_nodes": int(np.count_nonzero(committed_membership != membership)),
        "before_membership": before_stats,
        "proposed_membership": proposed_stats,
        "after_membership": after_stats,
        "trim_diagnostics": trim_diagnostics,
        "paths": paths,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary, committed_membership if status == "committed" else None


def _postprocess_run_status(
    *,
    committed_iterations: int,
    trim_committed: bool,
    stop_reason: str,
) -> str:
    if stop_reason in {
        "hard_cap_not_satisfied",
        "hard_cap_rolled_back_quality_below_threshold",
    }:
        return stop_reason
    if committed_iterations or trim_committed:
        return "committed"
    return "no_committed_iterations"


def _run_iterative_apply(
    graph,
    initial_membership: np.ndarray,
    node_weights: np.ndarray,
    gamma_multipliers: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    initial_phases: list[dict],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_membership = np.asarray(initial_membership, dtype=np.uint64).copy()
    original_membership = current_membership.copy()
    iterations: list[dict] = []
    all_phases = list(initial_phases)
    quality_before = float(
        graph.cpm_quality(membership=current_membership, resolution=args.resolution)
    )
    predicted_delta_q_sum_total = 0.0
    applied_delta_q_sum_total = 0.0
    changed_nodes_step_sum = 0
    stop_reason = "max_iterations_reached"
    trim_summary: dict | None = None

    for iteration in range(1, int(args.apply_iterations) + 1):
        before_stats = _membership_weight_summary(
            current_membership,
            node_weights,
            min_weight=float(args.target_min_doc_weight),
            max_weight=float(args.target_max_doc_weight),
        )
        candidate_clusters = _current_oversize_candidate_clusters(
            current_membership,
            node_weights,
            max_weight=float(args.target_max_doc_weight),
            max_candidates=int(args.max_candidates),
        )
        if candidate_clusters.size == 0:
            stop_reason = "no_current_oversize_candidates"
            break

        iteration_dir = output_dir / f"iteration_{iteration:03d}"
        iteration_phases: list[dict] = []
        _log(
            "iteration_start "
            f"{iteration} candidate_clusters={candidate_clusters.size} "
            f"max_doc_weight={before_stats['max_doc_weight']:.6g} "
            f"n_above_max={before_stats['n_above_max_doc_weight']}"
        )
        raw, selection_rows, probe_summary = _run_probe_selection_pass(
            graph,
            current_membership,
            candidate_clusters,
            gamma_multipliers,
            iteration_dir,
            args,
            iteration_phases,
        )
        apply_summary, applied_membership = _phase(
            "apply_split_repair_candidates",
            iteration_phases,
            lambda: _apply_selected_candidates(
                graph,
                current_membership,
                candidate_clusters,
                gamma_multipliers,
                selection_rows,
                iteration_dir,
                args,
                membership_output_path=iteration_dir
                / "split_repair_applied_membership.parquet",
            ),
        )
        probe_summary["apply"] = apply_summary
        probe_summary["paths"].update(apply_summary.get("paths", {}))
        probe_summary["phases"] = iteration_phases
        Path(probe_summary["paths"]["summary"]).write_text(
            json.dumps(probe_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        for phase in iteration_phases:
            phase["iteration"] = iteration
        all_phases.extend(iteration_phases)

        selected_rows = [row for row in selection_rows if row["selected_for_apply"]]
        after_stats = before_stats
        committed = apply_summary.get("status") == "committed"
        if committed:
            assert applied_membership is not None
            current_membership = applied_membership
            after_stats = _membership_weight_summary(
                current_membership,
                node_weights,
                min_weight=float(args.target_min_doc_weight),
                max_weight=float(args.target_max_doc_weight),
            )
            predicted_delta_q_sum_total += float(apply_summary["predicted_delta_q_sum"])
            applied_delta_q_sum_total += float(apply_summary["exact_delta_q"])
            changed_nodes_step_sum += int(apply_summary["changed_nodes"])

        iteration_record = {
            "iteration": iteration,
            "candidate_source": "current_oversize",
            "candidate_clusters": int(candidate_clusters.size),
            "candidate_cluster_ids": [int(x) for x in candidate_clusters.tolist()],
            "before_membership": before_stats,
            "after_membership": after_stats,
            "n_selected": int(apply_summary.get("n_selected", 0)),
            "n_applied": int(apply_summary.get("n_applied", 0)),
            "status": str(apply_summary.get("status", "")),
            "exact_delta_q": float(apply_summary.get("exact_delta_q", 0.0)),
            "predicted_delta_q_sum": float(
                apply_summary.get("predicted_delta_q_sum", 0.0)
            ),
            "changed_nodes": int(apply_summary.get("changed_nodes", 0)),
            "selected_rows": selected_rows,
            "paths": probe_summary["paths"],
        }
        iterations.append(iteration_record)
        _log(
            "iteration_done "
            f"{iteration} status={iteration_record['status']} "
            f"delta_q={iteration_record['exact_delta_q']:.6g} "
            f"max_doc_weight_after={after_stats['max_doc_weight']:.6g} "
            f"n_above_max_after={after_stats['n_above_max_doc_weight']}"
        )

        if not committed:
            stop_reason = str(apply_summary.get("status", "not_committed"))
            break
        if (
            float(args.target_max_doc_weight) > 0.0
            and after_stats["n_above_max_doc_weight"] == 0
        ):
            stop_reason = "target_max_satisfied"
            break

        # Subsequent iterations operate on the newly committed membership.
        del raw

    if args.apply_oversize_boundary_trim:
        _log("trim_start oversize_boundary_moves")
        trim_summary, trim_membership = _phase(
            "apply_oversize_boundary_trim",
            all_phases,
            lambda: _apply_oversize_boundary_trim(
                graph,
                current_membership,
                node_weights,
                output_dir,
                args,
                quality_floor=quality_before + float(args.apply_min_quality_delta),
            ),
        )
        if trim_summary.get("status") == "committed":
            assert trim_membership is not None
            current_membership = trim_membership
            final_trim_stats = _membership_weight_summary(
                current_membership,
                node_weights,
                min_weight=float(args.target_min_doc_weight),
                max_weight=float(args.target_max_doc_weight),
            )
            if final_trim_stats["n_above_max_doc_weight"] == 0:
                stop_reason = "target_max_satisfied_after_trim"
        _log(
            "trim_done "
            f"status={trim_summary.get('status')} "
            f"n_moves={trim_summary.get('n_moves', 0)} "
            f"delta_q={trim_summary.get('exact_delta_q', 0.0):.6g}"
        )

    final_stats = _membership_weight_summary(
        current_membership,
        node_weights,
        min_weight=float(args.target_min_doc_weight),
        max_weight=float(args.target_max_doc_weight),
    )
    initial_stats = _membership_weight_summary(
        original_membership,
        node_weights,
        min_weight=float(args.target_min_doc_weight),
        max_weight=float(args.target_max_doc_weight),
    )
    target_max_satisfied = (
        float(args.target_max_doc_weight) <= 0.0
        or final_stats["n_above_max_doc_weight"] == 0
    )
    if (
        str(args.oversize_acceptance_mode) == "hard_cap"
        and args.apply_oversize_boundary_trim
        and not target_max_satisfied
    ):
        if trim_summary and trim_summary.get("status") == "rolled_back_quality_below_threshold":
            stop_reason = "hard_cap_rolled_back_quality_below_threshold"
        else:
            stop_reason = "hard_cap_not_satisfied"
    quality_after = float(
        graph.cpm_quality(membership=current_membership, resolution=args.resolution)
    )
    committed_iterations = sum(1 for row in iterations if row["status"] == "committed")
    trim_committed = bool(trim_summary and trim_summary.get("status") == "committed")
    run_status = _postprocess_run_status(
        committed_iterations=committed_iterations,
        trim_committed=trim_committed,
        stop_reason=stop_reason,
    )
    trim_exact_delta_q = (
        float(trim_summary.get("exact_delta_q", 0.0)) if trim_committed else 0.0
    )
    final_exact_delta_q = quality_after - quality_before
    changed_nodes_vs_initial = int(
        np.count_nonzero(original_membership != current_membership)
    )
    paths = {
        "summary": str(output_dir / "iterative_split_repair_apply_summary.json"),
    }
    if committed_iterations or trim_committed:
        if run_status == "committed":
            final_membership_path = (
                args.applied_membership_output
                if args.applied_membership_output is not None
                else output_dir / "split_repair_applied_membership.parquet"
            )
            final_membership_path_key = "applied_membership"
        else:
            final_membership_path = output_dir / "split_repair_diagnostic_membership.parquet"
            final_membership_path_key = "diagnostic_membership"
        final_membership_path.parent.mkdir(parents=True, exist_ok=True)
        _write_membership(final_membership_path, current_membership)
        paths[final_membership_path_key] = str(final_membership_path)

    summary = {
        "status": run_status,
        "stop_reason": stop_reason,
        "n_iterations_requested": int(args.apply_iterations),
        "n_iterations_run": len(iterations),
        "n_committed_iterations": int(committed_iterations),
        "trim_committed": trim_committed,
        "quality_before": quality_before,
        "quality_after_final": quality_after,
        "exact_delta_q_total": final_exact_delta_q,
        "split_repair_exact_delta_q": applied_delta_q_sum_total,
        "trim_exact_delta_q": trim_exact_delta_q,
        "applied_exact_delta_q_sum": applied_delta_q_sum_total,
        "predicted_delta_q_sum_total": predicted_delta_q_sum_total,
        "changed_nodes_step_sum": int(changed_nodes_step_sum),
        "changed_nodes_vs_initial": changed_nodes_vs_initial,
        "target_max_satisfied": bool(target_max_satisfied),
        "initial_membership": initial_stats,
        "final_membership": final_stats,
        "iterations": iterations,
        "trim": trim_summary,
        "paths": paths,
        "phases": all_phases,
        "rss_mb_final": _rss_mb(),
        "hwm_mb_final": _hwm_mb(),
    }
    summary.update(
        _postprocess_transition_report(
            args,
            initial_stats,
            final_stats,
            changed_nodes=changed_nodes_vs_initial,
            split_repair_exact_delta_q=applied_delta_q_sum_total,
            trim_exact_delta_q=trim_exact_delta_q,
            final_exact_delta_q=final_exact_delta_q,
            stop_reason=stop_reason,
        )
    )
    Path(paths["summary"]).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


# Keep CLI-only argparse and file layout here, while sharing the reusable
# hierarchy postprocess policy calculations with build_hierarchy.
_cluster_weight_arrays = _hierarchy_postprocess._cluster_weight_arrays
_membership_weight_summary = _hierarchy_postprocess.membership_weight_summary
_oversize_residual_summary = _hierarchy_postprocess._oversize_residual_summary
_trim_source_move_counts = _hierarchy_postprocess._trim_source_move_counts
_trim_infeasibility_diagnostics = _hierarchy_postprocess._trim_infeasibility_diagnostics
_current_oversize_candidate_clusters = (
    _hierarchy_postprocess.current_oversize_candidate_clusters
)
_quality_floor_prefix_move_count = _hierarchy_postprocess._quality_floor_prefix_move_count
_trim_prefix_membership = _hierarchy_postprocess._trim_prefix_membership
_write_trim_move_rows = _hierarchy_postprocess._write_trim_move_rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--gamma-multipliers", default="1.02,1.05,1.10,1.15,1.20,1.25")
    parser.add_argument("--min-core-weight", type=float, default=25.0)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--repair-epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pair-seeded-probes",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--policy", default="")
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--target-min-doc-weight", type=float, default=0.0)
    parser.add_argument("--target-max-doc-weight", type=float, default=0.0)
    parser.add_argument(
        "--oversize-policy",
        choices=OVERSIZE_ACCEPTANCE_MODES,
        dest="oversize_acceptance_mode",
        default="quality_first",
        help=(
            "Postprocess policy for remaining oversize clusters. quality_first "
            "keeps trim moves non-negative by default; hard_cap permits a small "
            "negative trim delta by default while the final exact delta remains "
            "bounded by --apply-min-quality-delta."
        ),
    )
    parser.add_argument(
        "--oversize-acceptance-mode",
        choices=OVERSIZE_ACCEPTANCE_MODES,
        dest="oversize_acceptance_mode",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--selection-mode",
        choices=("utility_cost", "oversize_first"),
        default="oversize_first",
        help=(
            "Candidate ranking mode. oversize_first is the apply-mode default for "
            "clusters above --target-max-doc-weight; utility_cost keeps the older "
            "quality/cost diagnostic ranking."
        ),
    )
    parser.add_argument("--selection-singleton-budget", type=float, default=25.0)
    parser.add_argument("--selection-max-selected", type=int, default=0)
    parser.add_argument(
        "--apply-split-repair-candidates",
        action="store_true",
        help="Experimental: write a proposed membership for selected non-conflicting candidates.",
    )
    parser.add_argument(
        "--apply-iterations",
        type=int,
        default=1,
        help=(
            "Repeat apply mode over current oversize clusters. Values greater "
            "than 1 require --apply-split-repair-candidates and "
            "--target-max-doc-weight."
        ),
    )
    parser.add_argument("--applied-membership-output", type=Path, default=None)
    parser.add_argument("--apply-min-quality-delta", type=float, default=0.0)
    parser.add_argument(
        "--apply-oversize-boundary-trim",
        action="store_true",
        help=(
            "After split-repair apply iterations, greedily move delta-bounded "
            "boundary nodes from remaining oversize clusters without making "
            "target clusters exceed --target-max-doc-weight."
        ),
    )
    parser.add_argument("--trim-min-delta-q", type=float, default=None)
    parser.add_argument("--trim-max-moves-per-cluster", type=int, default=0)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    _normalize_args(parser, args)
    return args


def _normalize_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.trim_min_delta_q is None:
        if args.oversize_acceptance_mode == "hard_cap":
            args.trim_min_delta_q = HARD_CAP_DEFAULT_TRIM_MIN_DELTA_Q
        else:
            args.trim_min_delta_q = 0.0
        args.trim_min_delta_q_source = "mode_default"
    else:
        args.trim_min_delta_q_source = "explicit"

    if args.apply_iterations < 1:
        parser.error("--apply-iterations must be >= 1")
    if args.apply_iterations > 1 and not args.apply_split_repair_candidates:
        parser.error("--apply-iterations > 1 requires --apply-split-repair-candidates")
    if args.apply_iterations > 1 and args.target_max_doc_weight <= 0.0:
        parser.error("--apply-iterations > 1 requires --target-max-doc-weight > 0")
    if args.apply_oversize_boundary_trim and not args.apply_split_repair_candidates:
        parser.error("--apply-oversize-boundary-trim requires --apply-split-repair-candidates")
    if args.apply_oversize_boundary_trim and args.target_max_doc_weight <= 0.0:
        parser.error("--apply-oversize-boundary-trim requires --target-max-doc-weight > 0")
    if (
        args.oversize_acceptance_mode == "quality_first"
        and float(args.trim_min_delta_q) < 0.0
    ):
        parser.error("--trim-min-delta-q must be >= 0 in quality_first mode")
    if args.trim_max_moves_per_cluster < 0:
        parser.error("--trim-max-moves-per-cluster must be >= 0")


def main() -> None:
    args = _parse_args()

    src_path = args.graph_dir / "src.u32.bin"
    dst_path = args.graph_dir / "dst.u32.bin"
    weight_path = args.graph_dir / "weight.f64.bin"
    node_weights_path = args.graph_dir / "node_weights.f64.bin"
    n_nodes = node_weights_path.stat().st_size // np.dtype(np.float64).itemsize
    gamma_multipliers = np.asarray(
        [float(x) for x in args.gamma_multipliers.split(",") if x.strip()],
        dtype=np.float64,
    )
    phases: list[dict] = []

    _log(f"sciscape_leiden={sciscape_leiden.__file__}")
    candidate_clusters = _phase(
        "candidate_load",
        phases,
        lambda: _load_candidates(args.candidates, args.policy or None, args.max_candidates),
    )
    _log(f"candidate_clusters={candidate_clusters.size}")
    graph = _phase(
        "graph_load",
        phases,
        lambda: sciscape_leiden.load_graph_raw_files(
            int(n_nodes),
            str(src_path),
            str(dst_path),
            str(weight_path),
            str(node_weights_path),
        ),
    )
    membership = _phase("membership_load", phases, lambda: _load_membership(args.membership))
    node_weights: np.ndarray | None = None
    if args.apply_split_repair_candidates:
        node_weights = _phase(
            "node_weights_load",
            phases,
            lambda: _load_node_weights(node_weights_path),
        )

    if args.apply_split_repair_candidates and (
        args.apply_iterations > 1 or args.apply_oversize_boundary_trim
    ):
        summary = _run_iterative_apply(
            graph,
            membership,
            node_weights,
            gamma_multipliers,
            args.output_dir,
            args,
            phases,
        )
    else:
        raw, selection_rows, summary = _run_probe_selection_pass(
            graph,
            membership,
            candidate_clusters,
            gamma_multipliers,
            args.output_dir,
            args,
            phases,
        )

    if (
        args.apply_split_repair_candidates
        and args.apply_iterations == 1
        and not args.apply_oversize_boundary_trim
    ):
        apply_summary, applied_membership = _phase(
            "apply_split_repair_candidates",
            phases,
            lambda: _apply_selected_candidates(
                graph,
                membership,
                candidate_clusters,
                gamma_multipliers,
                selection_rows,
                args.output_dir,
                args,
            ),
        )
        summary["apply"] = apply_summary
        summary["paths"].update(apply_summary.get("paths", {}))
        summary["phases"] = phases
        assert node_weights is not None
        final_membership = membership
        split_repair_exact_delta_q = 0.0
        changed_nodes = 0
        if apply_summary.get("status") == "committed":
            assert applied_membership is not None
            final_membership = applied_membership
            split_repair_exact_delta_q = float(apply_summary.get("exact_delta_q", 0.0))
            changed_nodes = int(apply_summary.get("changed_nodes", 0))
        before_stats = _membership_weight_summary(
            membership,
            node_weights,
            min_weight=float(args.target_min_doc_weight),
            max_weight=float(args.target_max_doc_weight),
        )
        after_stats = _membership_weight_summary(
            final_membership,
            node_weights,
            min_weight=float(args.target_min_doc_weight),
            max_weight=float(args.target_max_doc_weight),
        )
        summary.update(
            _postprocess_transition_report(
                args,
                before_stats,
                after_stats,
                changed_nodes=changed_nodes,
                split_repair_exact_delta_q=split_repair_exact_delta_q,
                trim_exact_delta_q=0.0,
                final_exact_delta_q=split_repair_exact_delta_q,
                stop_reason=str(apply_summary.get("status", "")),
            )
        )
        Path(summary["paths"]["summary"]).write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    _log("summary_json_start")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    _log("summary_json_end")


if __name__ == "__main__":
    main()
