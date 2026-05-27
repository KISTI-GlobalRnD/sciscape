"""Probe cheap external-attachment grains for split-repair pre-screening."""

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
    "incident_directed_edges",
    "source_directed_edges",
    "external_directed_edges",
    "n_external_groups",
    "assigned_count",
    "assigned_weight",
    "assigned_fraction",
    "largest_group_target",
    "largest_group_count",
    "largest_group_weight",
    "largest_group_fraction",
    "largest_group_to_target_weight",
    "largest_group_cut_weight",
    "largest_group_move_delta_q",
    "largest_group_split_delta_q",
    "second_group_target",
    "second_group_weight",
    "second_group_fraction",
    "best_group_target",
    "best_group_count",
    "best_group_weight",
    "best_group_fraction",
    "best_group_to_target_weight",
    "best_group_cut_weight",
    "best_group_move_delta_q",
    "best_group_split_delta_q",
    "best_group_delta_q",
    "best_group_action",
    "positive_group_count",
    "positive_group_weight",
    "near_neutral_group_count",
    "near_neutral_group_weight",
    "priority_delta_per_incident_edge",
    "recommended_for_split_repair",
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


def _write_outputs(
    raw: dict[str, np.ndarray],
    output_dir: Path,
    phases: list[dict],
    args: argparse.Namespace,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    recommended = np.asarray(raw["recommended_for_split_repair"], dtype=bool)
    priority = np.asarray(raw["priority"], dtype=np.float64)
    arrays_path = output_dir / "external_grain_probes.npz"
    np.savez(arrays_path, **raw)

    order = np.lexsort((raw["cluster"], -priority, ~recommended))
    csv_path = output_dir / "external_grain_probes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            row = {field: raw[field][idx] for field in FIELDS if field in raw}
            row["rank"] = rank
            row["priority_delta_per_incident_edge"] = float(priority[idx])
            row["recommended_for_split_repair"] = bool(recommended[idx])
            for key, value in list(row.items()):
                if isinstance(value, np.generic):
                    row[key] = value.item()
            writer.writerow(row)

    summary_path = output_dir / "external_grain_probe_summary.json"
    positive = raw["best_group_delta_q"] > 0
    summary = {
        "n_probes": int(raw["cluster"].shape[0]),
        "n_positive_best_group": int(positive.sum()),
        "n_recommended_for_split_repair": int(recommended.sum()),
        "best_group_delta_q": {
            "p50": _percentile(raw["best_group_delta_q"], 50),
            "p90": _percentile(raw["best_group_delta_q"], 90),
            "p95": _percentile(raw["best_group_delta_q"], 95),
            "p99": _percentile(raw["best_group_delta_q"], 99),
            "max": float(raw["best_group_delta_q"].max())
            if raw["best_group_delta_q"].size
            else 0.0,
        },
        "assigned_fraction": {
            "p50": _percentile(raw["assigned_fraction"], 50),
            "p90": _percentile(raw["assigned_fraction"], 90),
            "p95": _percentile(raw["assigned_fraction"], 95),
        },
        "best_group_fraction": {
            "p50": _percentile(raw["best_group_fraction"], 50),
            "p90": _percentile(raw["best_group_fraction"], 90),
            "p95": _percentile(raw["best_group_fraction"], 95),
        },
        "incident_directed_edges": {
            "p50": _percentile(raw["incident_directed_edges"], 50),
            "p90": _percentile(raw["incident_directed_edges"], 90),
            "p95": _percentile(raw["incident_directed_edges"], 95),
            "max": int(raw["incident_directed_edges"].max())
            if raw["incident_directed_edges"].size
            else 0,
        },
        "recommendation_thresholds": {
            "min_doc_weight": args.min_doc_weight,
            "max_incident_directed_edges": args.max_incident_directed_edges,
            "min_best_delta_q": args.min_best_delta_q,
            "min_assigned_fraction": args.min_assigned_fraction,
            "min_best_group_fraction": args.min_best_group_fraction,
        },
        "phases": phases,
        "paths": {
            "arrays": str(arrays_path),
            "probes": str(csv_path),
            "summary": str(summary_path),
        },
        "rss_mb_final": _rss_mb(),
        "hwm_mb_final": _hwm_mb(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--policy", default="")
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--min-doc-weight", type=float, default=0.0)
    parser.add_argument("--max-incident-directed-edges", type=int, default=0)
    parser.add_argument("--min-best-delta-q", type=float, default=0.0)
    parser.add_argument("--min-assigned-fraction", type=float, default=0.0)
    parser.add_argument("--min-best-group-fraction", type=float, default=0.0)
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
        "external_grain_probes",
        phases,
        lambda: graph.external_grain_probes(
            membership=membership,
            candidate_clusters=candidate_clusters,
            resolution=args.resolution,
            epsilon=args.epsilon,
            min_doc_weight=args.min_doc_weight,
            max_incident_directed_edges=args.max_incident_directed_edges,
            min_best_delta_q=args.min_best_delta_q,
            min_assigned_fraction=args.min_assigned_fraction,
            min_best_group_fraction=args.min_best_group_fraction,
        ),
    )
    raw = {key: np.asarray(value) for key, value in raw.items()}
    summary = _phase("write_outputs", phases, lambda: _write_outputs(raw, args.output_dir, phases, args))
    _log("summary_json_start")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    _log("summary_json_end")


if __name__ == "__main__":
    main()
