"""Compare optimal cut against threshold-based cuts on the same dendrogram."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from _cut_common import (
    default_cut_min_size,
    prepare_hybrid_cut_context,
    project_contracted_membership,
    select_threshold_candidates,
    threshold_cut,
)

from sciscape.clustering.constrained_cut import constrained_cut
from sciscape.evaluation.stability import compute_quality_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut strategy ablation on the contracted dendrogram")
    parser.add_argument("edge_path", type=Path)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--nano-min-size", type=int, default=30)
    parser.add_argument("--cut-min-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dendrogram-mode", type=str, default="cpm", choices=["cpm", "triadic_cpm"])
    parser.add_argument("--n-thresholds", type=int, default=24, help="Number of threshold candidates to evaluate")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ctx = prepare_hybrid_cut_context(
        args.edge_path,
        target_pct=args.target_pct,
        nano_min_size=args.nano_min_size,
        seed=args.seed,
        dendrogram_mode=args.dendrogram_mode,
    )
    cut_min_size = args.cut_min_size or default_cut_min_size(ctx["nano_size_arr"])

    optimal = constrained_cut(
        ctx["linkage"],
        min_size=cut_min_size,
        leaf_sizes=ctx["nano_size_arr"],
    )
    optimal_membership = optimal.membership.astype(np.int64) if optimal.feasible else np.zeros(ctx["n_contracted"], dtype=np.int64)
    optimal_projected = project_contracted_membership(ctx["compact_membership"], optimal_membership)
    optimal_qr = compute_quality_report(
        ctx["edges_gcc"],
        optimal_projected,
        gamma=ctx["nano"].gamma,
        target_pct=args.target_pct,
    )

    threshold_runs: list[dict] = []
    best_threshold: dict | None = None
    for threshold in select_threshold_candidates(ctx["linkage"], n_thresholds=args.n_thresholds):
        result = threshold_cut(
            ctx["linkage"],
            threshold=threshold,
            min_size=cut_min_size,
            leaf_sizes=ctx["nano_size_arr"],
        )
        if result["feasible"]:
            projected = project_contracted_membership(
                ctx["compact_membership"],
                result["membership"],
            )
            qr = compute_quality_report(
                ctx["edges_gcc"],
                projected,
                gamma=ctx["nano"].gamma,
                target_pct=args.target_pct,
            )
            record = {
                "threshold": threshold,
                "feasible": True,
                "contracted_clusters": result["n_clusters"],
                "n_clusters": int((np.bincount(projected) > 0).sum()),
                "max_pct": qr.max_cluster_pct,
                "singleton_pct": qr.singleton_pct,
                "top5": qr.top5_sizes,
                "total_stability": result["total_stability"],
            }
        else:
            record = {
                "threshold": threshold,
                "feasible": False,
                "contracted_clusters": result["n_clusters"],
                "n_clusters": None,
                "max_pct": None,
                "singleton_pct": None,
                "top5": None,
                "total_stability": result["total_stability"],
            }
        threshold_runs.append(record)
        if record["feasible"]:
            if best_threshold is None:
                best_threshold = record
            elif record["n_clusters"] > best_threshold["n_clusters"]:
                best_threshold = record
            elif record["n_clusters"] == best_threshold["n_clusters"] and record["total_stability"] > best_threshold["total_stability"]:
                best_threshold = record

    elapsed = time.perf_counter() - t0
    payload = {
        "field": args.field,
        "method": "cut_ablation",
        "edge_path": str(args.edge_path),
        "target_pct": args.target_pct,
        "nano_min_size": args.nano_min_size,
        "cut_min_size": cut_min_size,
        "seed": args.seed,
        "dendrogram_mode": args.dendrogram_mode,
        "n_thresholds": args.n_thresholds,
        "elapsed": round(elapsed, 1),
        "n_nodes": len(ctx["gcc_uids"]),
        "n_edges": ctx["edges_gcc"].height,
        "n_contracted_nodes": ctx["n_contracted"],
        "nano": {
            "gamma": ctx["nano"].gamma,
            "n_clusters": ctx["nano"].n_clusters,
            "max_pct": ctx["nano"].max_pct,
            "singleton_pct": ctx["nano_qr"].singleton_pct,
        },
        "optimal_cut": {
            "feasible": optimal.feasible,
            "contracted_clusters": optimal.n_clusters,
            "n_clusters": int((np.bincount(optimal_projected) > 0).sum()),
            "max_pct": optimal_qr.max_cluster_pct,
            "singleton_pct": optimal_qr.singleton_pct,
            "top5": optimal_qr.top5_sizes,
            "total_stability": optimal.total_stability,
        },
        "threshold_candidates": threshold_runs,
        "best_threshold_cut": best_threshold,
    }
    out_path = args.output / f"{args.field}_cut_ablation.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    log.info("Optimal cut: %d clusters, max=%.1f%%", payload["optimal_cut"]["n_clusters"], payload["optimal_cut"]["max_pct"])
    if best_threshold:
        log.info(
            "Best threshold cut: threshold=%.6f, clusters=%d, max=%.1f%%",
            best_threshold["threshold"],
            best_threshold["n_clusters"],
            best_threshold["max_pct"],
        )
    else:
        log.info("Best threshold cut: no feasible threshold found")
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
