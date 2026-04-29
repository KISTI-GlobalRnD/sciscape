"""Probe high-gamma induced multi-core splits for adaptive refinement."""

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
    "internal_weight",
    "induced_directed_edges",
    "n_parts",
    "non_singleton_parts",
    "singleton_parts",
    "singleton_weight",
    "core_part_count",
    "core_part_weight",
    "largest_part_weight",
    "second_part_weight",
    "largest_part_fraction",
    "cut_weight",
    "split_delta_q_base",
    "split_delta_q_probe",
    "hysteresis_only",
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
    split = raw["n_parts"] > 1
    base_positive = raw["split_delta_q_base"] > 0
    probe_positive = raw["split_delta_q_probe"] > 0
    meaningful_core = raw["core_part_count"] >= 2
    return {
        "n_probes": int(raw["cluster"].shape[0]),
        "n_split": int(split.sum()),
        "n_base_positive": int(base_positive.sum()),
        "n_probe_positive": int(probe_positive.sum()),
        "n_hysteresis_only": int(raw["hysteresis_only"].sum()),
        "n_meaningful_core_split": int((split & meaningful_core).sum()),
        "n_base_positive_meaningful_core_split": int((base_positive & meaningful_core).sum()),
        "n_probe_positive_meaningful_core_split": int((probe_positive & meaningful_core).sum()),
        "n_parts": {
            "p50": _percentile(raw["n_parts"].astype(np.float64), 50),
            "p90": _percentile(raw["n_parts"].astype(np.float64), 90),
            "p95": _percentile(raw["n_parts"].astype(np.float64), 95),
            "p99": _percentile(raw["n_parts"].astype(np.float64), 99),
            "max": int(raw["n_parts"].max()) if raw["n_parts"].size else 0,
        },
        "core_part_count": {
            "p50": _percentile(raw["core_part_count"].astype(np.float64), 50),
            "p90": _percentile(raw["core_part_count"].astype(np.float64), 90),
            "p95": _percentile(raw["core_part_count"].astype(np.float64), 95),
            "p99": _percentile(raw["core_part_count"].astype(np.float64), 99),
            "max": int(raw["core_part_count"].max()) if raw["core_part_count"].size else 0,
        },
        "split_delta_q_base": {
            "p50": _percentile(raw["split_delta_q_base"], 50),
            "p90": _percentile(raw["split_delta_q_base"], 90),
            "p95": _percentile(raw["split_delta_q_base"], 95),
            "p99": _percentile(raw["split_delta_q_base"], 99),
            "max": float(raw["split_delta_q_base"].max()) if raw["split_delta_q_base"].size else 0.0,
        },
        "split_delta_q_probe": {
            "p50": _percentile(raw["split_delta_q_probe"], 50),
            "p90": _percentile(raw["split_delta_q_probe"], 90),
            "p95": _percentile(raw["split_delta_q_probe"], 95),
            "p99": _percentile(raw["split_delta_q_probe"], 99),
            "max": float(raw["split_delta_q_probe"].max()) if raw["split_delta_q_probe"].size else 0.0,
        },
        "largest_part_fraction": {
            "p50": _percentile(raw["largest_part_fraction"], 50),
            "p90": _percentile(raw["largest_part_fraction"], 90),
            "p95": _percentile(raw["largest_part_fraction"], 95),
        },
        "singleton_weight": {
            "p50": _percentile(raw["singleton_weight"], 50),
            "p90": _percentile(raw["singleton_weight"], 90),
            "p95": _percentile(raw["singleton_weight"], 95),
        },
        "phases": phases,
        "paths": paths,
        "rss_mb_final": _rss_mb(),
        "hwm_mb_final": _hwm_mb(),
    }


def _write_outputs(raw: dict[str, np.ndarray], output_dir: Path, phases: list[dict]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "multi_core_split_probes.npz"
    np.savez(arrays_path, **raw)

    order = np.lexsort(
        (
            raw["cluster"],
            -raw["split_delta_q_probe"],
            -raw["split_delta_q_base"],
        )
    )
    csv_path = output_dir / "multi_core_split_probes.csv"
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
                    "internal_weight": float(raw["internal_weight"][idx]),
                    "induced_directed_edges": int(raw["induced_directed_edges"][idx]),
                    "n_parts": int(raw["n_parts"][idx]),
                    "non_singleton_parts": int(raw["non_singleton_parts"][idx]),
                    "singleton_parts": int(raw["singleton_parts"][idx]),
                    "singleton_weight": float(raw["singleton_weight"][idx]),
                    "core_part_count": int(raw["core_part_count"][idx]),
                    "core_part_weight": float(raw["core_part_weight"][idx]),
                    "largest_part_weight": float(raw["largest_part_weight"][idx]),
                    "second_part_weight": float(raw["second_part_weight"][idx]),
                    "largest_part_fraction": float(raw["largest_part_fraction"][idx]),
                    "cut_weight": float(raw["cut_weight"][idx]),
                    "split_delta_q_base": float(raw["split_delta_q_base"][idx]),
                    "split_delta_q_probe": float(raw["split_delta_q_probe"][idx]),
                    "hysteresis_only": bool(raw["hysteresis_only"][idx]),
                }
            )
    paths = {"arrays": str(arrays_path), "probes": str(csv_path)}
    summary = _summary(raw, phases, paths)
    summary_path = output_dir / "multi_core_split_probe_summary.json"
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
    parser.add_argument("--gamma-multipliers", default="1.25,1.5,2.0,3.0")
    parser.add_argument("--min-core-weight", type=float, default=25.0)
    parser.add_argument("--randomness", type=float, default=0.01)
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
        "multi_core_split_probes",
        phases,
        lambda: graph.multi_core_split_probes(
            membership=membership,
            candidate_clusters=candidate_clusters,
            resolution=args.resolution,
            gamma_multipliers=gamma_multipliers,
            min_core_weight=args.min_core_weight,
            randomness=args.randomness,
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
