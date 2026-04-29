"""Probe forced high-gamma splits followed by baseline-gamma merge repair."""

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
    "gamma_multiplier",
    "probe_resolution",
    "block_count",
    "doc_weight",
    "n_parts",
    "core_part_count",
    "singleton_weight",
    "cut_weight",
    "split_delta_q_base",
    "split_delta_q_probe",
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


def _write_outputs(raw: dict[str, np.ndarray], output_dir: Path, phases: list[dict]) -> dict:
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
                    "n_parts": int(raw["n_parts"][idx]),
                    "core_part_count": int(raw["core_part_count"][idx]),
                    "singleton_weight": float(raw["singleton_weight"][idx]),
                    "cut_weight": float(raw["cut_weight"][idx]),
                    "split_delta_q_base": float(raw["split_delta_q_base"][idx]),
                    "split_delta_q_probe": float(raw["split_delta_q_probe"][idx]),
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
    paths = {"arrays": str(arrays_path), "probes": str(csv_path)}
    summary = _summary(raw, phases, paths)
    summary_path = output_dir / "split_merge_repair_probe_summary.json"
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
    parser.add_argument("--gamma-multipliers", default="1.02,1.05,1.10,1.15,1.20,1.25")
    parser.add_argument("--min-core-weight", type=float, default=25.0)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--repair-epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default="")
    parser.add_argument("--max-candidates", type=int, default=1000)
    args = parser.parse_args()

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
    raw = _phase(
        "split_merge_repair_probes",
        phases,
        lambda: graph.split_merge_repair_probes(
            membership=membership,
            candidate_clusters=candidate_clusters,
            resolution=args.resolution,
            gamma_multipliers=gamma_multipliers,
            min_core_weight=args.min_core_weight,
            randomness=args.randomness,
            repair_epsilon=args.repair_epsilon,
            seed=args.seed,
        ),
    )
    raw = {key: np.asarray(value) for key, value in raw.items()}
    summary = _phase("write_outputs", phases, lambda: _write_outputs(raw, args.output_dir, phases))
    _log("summary_json_start")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    _log("summary_json_end")


if __name__ == "__main__":
    main()
