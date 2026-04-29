"""Run adaptive boundary-candidate diagnostics on a cached Rust graph sidecar.

This script intentionally depends only on the native ``sciscape_leiden`` module
and common array/parquet packages so it can be copied to GPU workers without
installing the full local SciScape checkout.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import sciscape_leiden


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
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _run_phase(name: str, phases: list[dict], fn):
    _log(f"phase_start {name} rss={_rss_mb():.1f}MB hwm={_hwm_mb():.1f}MB")
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    phase = {
        "name": name,
        "elapsed_sec": elapsed,
        "rss_mb": _rss_mb(),
        "hwm_mb": _hwm_mb(),
    }
    phases.append(phase)
    _log(
        f"phase_done {name} elapsed={elapsed:.2f}s "
        f"rss={phase['rss_mb']:.1f}MB hwm={phase['hwm_mb']:.1f}MB"
    )
    return result


def _load_membership(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["node_idx", "cluster"])
    node_idx = table.column("node_idx").combine_chunks().to_numpy(zero_copy_only=False)
    cluster = table.column("cluster").combine_chunks().to_numpy(zero_copy_only=False)
    order = np.argsort(node_idx)
    return np.asarray(cluster[order], dtype=np.uint64)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50_000)
    args = parser.parse_args()

    src_path = args.graph_dir / "src.u32.bin"
    dst_path = args.graph_dir / "dst.u32.bin"
    weight_path = args.graph_dir / "weight.f64.bin"
    node_weights_path = args.graph_dir / "node_weights.f64.bin"
    n_nodes = node_weights_path.stat().st_size // np.dtype(np.float64).itemsize
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phases: list[dict] = []
    _log(f"sciscape_leiden={sciscape_leiden.__file__}")
    graph = _run_phase(
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
    membership = _run_phase(
        "membership_load",
        phases,
        lambda: _load_membership(args.membership),
    )
    stats = _run_phase(
        "cluster_graph_stats",
        phases,
        lambda: graph.cluster_graph_stats(
            membership,
            args.resolution,
            args.min_weight,
            args.max_weight,
            args.top_k,
        ),
    )

    arrays_path = args.output_dir / "cluster_graph_stats.npz"

    def write_stats() -> dict:
        np.savez(
            arrays_path,
            block_count=stats["block_count"],
            doc_weight=stats["doc_weight"],
            internal_weight=stats["internal_weight"],
            external_weight=stats["external_weight"],
            degree=stats["degree"],
            top_neighbor=stats["top_neighbor"],
            top_neighbor_weight=stats["top_neighbor_weight"],
            second_neighbor=stats["second_neighbor"],
            second_neighbor_weight=stats["second_neighbor_weight"],
            neighbor_weight_ratio=stats["neighbor_weight_ratio"],
            conductance=stats["conductance"],
            leafness=stats["leafness"],
            band_distance=stats["band_distance"],
            candidate_source=stats["candidate_source"],
            candidate_target=stats["candidate_target"],
            candidate_edge_weight=stats["candidate_edge_weight"],
            candidate_delta_q=stats["candidate_delta_q"],
            candidate_merged_weight=stats["candidate_merged_weight"],
            candidate_size_band_gain=stats["candidate_size_band_gain"],
        )
        active = np.asarray(stats["block_count"]) > 0
        with_second = np.asarray(stats["second_neighbor"]) >= 0
        summary = {
            "dataset": args.output_dir.name,
            "resolution": args.resolution,
            "min_weight": args.min_weight,
            "max_weight": args.max_weight,
            "top_k": args.top_k,
            "n_clusters": int(len(stats["block_count"])),
            "n_active_clusters": int(active.sum()),
            "n_merge_candidates": int(len(stats["candidate_source"])),
            "n_with_second_neighbor": int((active & with_second).sum()),
            "neighbor_weight_ratio": {
                "p50": _percentile(stats["neighbor_weight_ratio"][active], 50),
                "p90": _percentile(stats["neighbor_weight_ratio"][active], 90),
                "p95": _percentile(stats["neighbor_weight_ratio"][active], 95),
                "p99": _percentile(stats["neighbor_weight_ratio"][active], 99),
            },
            "conductance": {
                "p50": _percentile(stats["conductance"][active], 50),
                "p90": _percentile(stats["conductance"][active], 90),
                "p95": _percentile(stats["conductance"][active], 95),
                "p99": _percentile(stats["conductance"][active], 99),
            },
            "paths": {"cluster_arrays": str(arrays_path)},
            "phases": phases,
            "rss_mb_final": _rss_mb(),
            "hwm_mb_final": _hwm_mb(),
        }
        summary_path = args.output_dir / "boundary_stats_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary

    summary = _run_phase("write_stats", phases, write_stats)
    summary["phases"] = phases
    (args.output_dir / "boundary_stats_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _log("summary_json_start")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    _log("summary_json_end")


if __name__ == "__main__":
    main()
