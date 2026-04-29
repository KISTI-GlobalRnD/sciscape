"""Probe block-level moves for adaptive boundary refinement candidates."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import sciscape_leiden


FIELDS = [
    "rank",
    "cluster",
    "block_count",
    "doc_weight",
    "internal_weight",
    "external_weight",
    "conductance",
    "leafness",
    "top_neighbor",
    "top_neighbor_weight",
    "second_neighbor",
    "second_neighbor_weight",
    "neighbor_weight_ratio",
    "positive_move_count",
    "positive_move_weight",
    "positive_delta_q",
    "near_neutral_move_count",
    "near_neutral_move_weight",
    "near_neutral_delta_q",
    "best_move_delta_q",
    "best_move_node",
    "best_move_target",
    "top_move_count",
    "second_move_count",
]


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


def _summary(raw: dict[str, np.ndarray], phases: list[dict], epsilon: float, paths: dict[str, str]) -> dict:
    positive = raw["positive_move_count"] > 0
    near = raw["near_neutral_move_count"] > 0
    return {
        "epsilon": epsilon,
        "n_probes": int(raw["cluster"].shape[0]),
        "n_with_positive_moves": int(positive.sum()),
        "n_with_near_neutral_moves": int(near.sum()),
        "total_positive_move_count": int(raw["positive_move_count"].sum()),
        "total_positive_move_weight": float(raw["positive_move_weight"].sum()),
        "total_positive_delta_q": float(raw["positive_delta_q"].sum()),
        "total_near_neutral_move_count": int(raw["near_neutral_move_count"].sum()),
        "total_near_neutral_move_weight": float(raw["near_neutral_move_weight"].sum()),
        "total_near_neutral_delta_q": float(raw["near_neutral_delta_q"].sum()),
        "best_move_delta_q": {
            "p50": _percentile(raw["best_move_delta_q"], 50),
            "p90": _percentile(raw["best_move_delta_q"], 90),
            "p95": _percentile(raw["best_move_delta_q"], 95),
            "p99": _percentile(raw["best_move_delta_q"], 99),
            "max": float(raw["best_move_delta_q"].max()) if raw["best_move_delta_q"].size else 0.0,
        },
        "positive_move_count": {
            "p50": _percentile(raw["positive_move_count"].astype(np.float64), 50),
            "p90": _percentile(raw["positive_move_count"].astype(np.float64), 90),
            "p95": _percentile(raw["positive_move_count"].astype(np.float64), 95),
            "p99": _percentile(raw["positive_move_count"].astype(np.float64), 99),
        },
        "phases": phases,
        "paths": paths,
        "rss_mb_final": _rss_mb(),
        "hwm_mb_final": _hwm_mb(),
    }


def _write_outputs(raw: dict[str, np.ndarray], output_dir: Path, phases: list[dict], epsilon: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "boundary_move_probes.npz"
    np.savez(arrays_path, **raw)

    order = np.argsort(-raw["best_move_delta_q"], kind="mergesort")
    csv_path = output_dir / "boundary_move_probes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "cluster": int(raw["cluster"][idx]),
                    "block_count": int(raw["block_count"][idx]),
                    "doc_weight": float(raw["doc_weight"][idx]),
                    "internal_weight": float(raw["internal_weight"][idx]),
                    "external_weight": float(raw["external_weight"][idx]),
                    "conductance": float(raw["conductance"][idx]),
                    "leafness": float(raw["leafness"][idx]),
                    "top_neighbor": int(raw["top_neighbor"][idx]),
                    "top_neighbor_weight": float(raw["top_neighbor_weight"][idx]),
                    "second_neighbor": int(raw["second_neighbor"][idx]),
                    "second_neighbor_weight": float(raw["second_neighbor_weight"][idx]),
                    "neighbor_weight_ratio": float(raw["neighbor_weight_ratio"][idx]),
                    "positive_move_count": int(raw["positive_move_count"][idx]),
                    "positive_move_weight": float(raw["positive_move_weight"][idx]),
                    "positive_delta_q": float(raw["positive_delta_q"][idx]),
                    "near_neutral_move_count": int(raw["near_neutral_move_count"][idx]),
                    "near_neutral_move_weight": float(raw["near_neutral_move_weight"][idx]),
                    "near_neutral_delta_q": float(raw["near_neutral_delta_q"][idx]),
                    "best_move_delta_q": float(raw["best_move_delta_q"][idx]),
                    "best_move_node": int(raw["best_move_node"][idx]),
                    "best_move_target": int(raw["best_move_target"][idx]),
                    "top_move_count": int(raw["top_move_count"][idx]),
                    "second_move_count": int(raw["second_move_count"][idx]),
                }
            )

    paths = {"arrays": str(arrays_path), "probes": str(csv_path)}
    summary = _summary(raw, phases, epsilon, paths)
    summary_path = output_dir / "boundary_move_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"] = str(summary_path)
    summary["paths"] = paths
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--policy", default="")
    parser.add_argument("--max-candidates", type=int, default=1000)
    args = parser.parse_args()

    src_path = args.graph_dir / "src.u32.bin"
    dst_path = args.graph_dir / "dst.u32.bin"
    weight_path = args.graph_dir / "weight.f64.bin"
    node_weights_path = args.graph_dir / "node_weights.f64.bin"
    n_nodes = node_weights_path.stat().st_size // np.dtype(np.float64).itemsize
    phases: list[dict] = []

    _log(f"sciscape_leiden={sciscape_leiden.__file__}")
    candidate_clusters = _phase(
        "candidate_load",
        phases,
        lambda: _load_candidates(
            args.candidates,
            args.policy or None,
            args.max_candidates,
        ),
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
    raw = _phase(
        "boundary_move_probes",
        phases,
        lambda: graph.boundary_move_probes(
            membership=membership,
            candidate_clusters=candidate_clusters,
            resolution=args.resolution,
            epsilon=args.epsilon,
        ),
    )
    raw = {key: np.asarray(value) for key, value in raw.items()}
    summary = _phase(
        "write_outputs",
        phases,
        lambda: _write_outputs(raw, args.output_dir, phases, args.epsilon),
    )
    _log("summary_json_start")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    _log("summary_json_end")


if __name__ == "__main__":
    main()
