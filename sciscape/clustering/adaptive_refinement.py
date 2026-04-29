"""Adaptive refinement diagnostics built on the Rust Leiden backend.

The functions here are observational. They summarize a baseline Leiden
membership and macro-merge dry-run candidates without changing the partition.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .leiden_rust import RustClusterGraphStats


def _percentiles(values: np.ndarray, percentiles: list[int]) -> dict[str, float]:
    if values.size == 0:
        return {f"p{p}": 0.0 for p in percentiles}
    return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}


def summarize_cluster_graph_stats(
    stats: RustClusterGraphStats,
    *,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
) -> dict[str, Any]:
    """Return compact aggregate diagnostics for a cluster-graph stats object."""

    active = stats.block_count > 0
    active_weight = stats.doc_weight[active]
    active_internal = stats.internal_weight[active]
    active_external = stats.external_weight[active]
    active_conductance = stats.conductance[active]
    active_leafness = stats.leafness[active]
    candidates = stats.candidate_source.shape[0]
    positive_delta = stats.candidate_delta_q > 0
    band_improving = stats.candidate_size_band_gain > 0

    summary: dict[str, Any] = {
        "n_clusters": int(stats.n_clusters),
        "n_active_clusters": int(active.sum()),
        "total_doc_weight": float(active_weight.sum()),
        "total_internal_weight": float(active_internal.sum()),
        "total_external_weight": float(active_external.sum()),
        "doc_weight": _percentiles(active_weight, [50, 90, 95, 99]),
        "conductance": _percentiles(active_conductance, [50, 90, 95, 99]),
        "leafness": _percentiles(active_leafness, [50, 90, 95, 99]),
        "n_merge_candidates": int(candidates),
        "n_positive_delta_candidates": int(positive_delta.sum()),
        "n_band_improving_candidates": int(band_improving.sum()),
        "n_positive_and_band_improving_candidates": int((positive_delta & band_improving).sum()),
    }
    if min_weight > 0:
        summary["n_below_min_weight"] = int((active_weight < min_weight).sum())
    if max_weight > 0:
        summary["n_above_max_weight"] = int((active_weight > max_weight).sum())
    if min_weight > 0 and max_weight > 0:
        summary["n_within_weight_band"] = int(
            ((active_weight >= min_weight) & (active_weight <= max_weight)).sum()
        )
    return summary


def write_adaptive_refinement_report(
    stats: RustClusterGraphStats,
    output_dir: Path,
    *,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
    top_candidates: int = 1000,
    write_cluster_arrays: bool = True,
) -> dict[str, str]:
    """Write JSON/CSV/NPZ diagnostics for adaptive-refinement dry-runs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_cluster_graph_stats(
        stats,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    summary_path = output_dir / "cluster_graph_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    candidate_path = output_dir / "macro_merge_candidates.csv"
    n_candidates = min(int(top_candidates), stats.n_candidates)
    with candidate_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "source",
                "target",
                "source_doc_weight",
                "target_doc_weight",
                "source_block_count",
                "target_block_count",
                "source_internal_weight",
                "target_internal_weight",
                "source_external_weight",
                "target_external_weight",
                "source_degree",
                "target_degree",
                "source_conductance",
                "target_conductance",
                "source_leafness",
                "target_leafness",
                "edge_weight",
                "delta_q",
                "merged_weight",
                "size_band_gain",
            ],
        )
        writer.writeheader()
        for idx in range(n_candidates):
            source = int(stats.candidate_source[idx])
            target = int(stats.candidate_target[idx])
            writer.writerow(
                {
                    "rank": idx + 1,
                    "source": source,
                    "target": target,
                    "source_doc_weight": float(stats.doc_weight[source]),
                    "target_doc_weight": float(stats.doc_weight[target]),
                    "source_block_count": int(stats.block_count[source]),
                    "target_block_count": int(stats.block_count[target]),
                    "source_internal_weight": float(stats.internal_weight[source]),
                    "target_internal_weight": float(stats.internal_weight[target]),
                    "source_external_weight": float(stats.external_weight[source]),
                    "target_external_weight": float(stats.external_weight[target]),
                    "source_degree": int(stats.degree[source]),
                    "target_degree": int(stats.degree[target]),
                    "source_conductance": float(stats.conductance[source]),
                    "target_conductance": float(stats.conductance[target]),
                    "source_leafness": float(stats.leafness[source]),
                    "target_leafness": float(stats.leafness[target]),
                    "edge_weight": float(stats.candidate_edge_weight[idx]),
                    "delta_q": float(stats.candidate_delta_q[idx]),
                    "merged_weight": float(stats.candidate_merged_weight[idx]),
                    "size_band_gain": float(stats.candidate_size_band_gain[idx]),
                }
            )

    paths = {
        "summary": str(summary_path),
        "merge_candidates": str(candidate_path),
    }
    if write_cluster_arrays:
        arrays_path = output_dir / "cluster_graph_stats.npz"
        np.savez(
            arrays_path,
            block_count=stats.block_count,
            doc_weight=stats.doc_weight,
            internal_weight=stats.internal_weight,
            external_weight=stats.external_weight,
            degree=stats.degree,
            top_neighbor=stats.top_neighbor,
            top_neighbor_weight=stats.top_neighbor_weight,
            conductance=stats.conductance,
            leafness=stats.leafness,
            band_distance=stats.band_distance,
        )
        paths["cluster_arrays"] = str(arrays_path)
    return paths


__all__ = [
    "summarize_cluster_graph_stats",
    "write_adaptive_refinement_report",
]
