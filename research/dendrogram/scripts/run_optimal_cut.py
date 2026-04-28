"""Run the hybrid nano->contract->dendrogram->optimal-cut pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from _cut_common import (
    default_cut_min_size,
    prepare_hybrid_cut_context,
    project_contracted_membership,
)

from sciscape.clustering.constrained_cut import constrained_cut
from sciscape.evaluation.stability import compute_quality_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid micro-cut with constrained_cut")
    parser.add_argument("edge_path", type=Path)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--nano-min-size", type=int, default=30)
    parser.add_argument(
        "--cut-min-size",
        type=int,
        default=None,
        help="Minimum original-node size for the cut. Default follows landscape heuristic.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dendrogram-mode", type=str, default="cpm", choices=["cpm", "triadic_cpm"])
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
    log.info("GCC: %d nodes, %d edges", ctx["graph"].vcount(), ctx["graph"].ecount())
    log.info(
        "Nano: gamma=%.2e, clusters=%d, max=%.1f%%",
        ctx["nano"].gamma,
        ctx["nano"].n_clusters,
        ctx["nano"].max_pct,
    )

    cut_min_size = args.cut_min_size or default_cut_min_size(ctx["nano_size_arr"])
    cut = constrained_cut(
        ctx["linkage"],
        min_size=cut_min_size,
        leaf_sizes=ctx["nano_size_arr"],
    )

    if not cut.feasible or cut.n_clusters <= 1:
        log.warning("Cut infeasible or trivial (%d clusters)", cut.n_clusters)
        cut_membership_contracted = np.zeros(ctx["n_contracted"], dtype=np.int64)
    else:
        cut_membership_contracted = cut.membership.astype(np.int64)

    micro_membership = project_contracted_membership(
        ctx["compact_membership"],
        cut_membership_contracted,
    )
    elapsed = time.perf_counter() - t0

    micro_qr = compute_quality_report(
        ctx["edges_gcc"],
        micro_membership,
        gamma=ctx["nano"].gamma,
        target_pct=args.target_pct,
    )

    membership_df = pl.DataFrame(
        {
            "uid": ctx["gcc_uids"],
            "cluster_nano": ctx["graph_membership"],
            "cluster_micro_cut": micro_membership.tolist(),
        }
    )
    membership_path = args.output / f"{args.field}_hybrid_cut_membership.parquet"
    membership_df.write_parquet(membership_path)

    payload = {
        "field": args.field,
        "method": "hybrid_optimal_cut",
        "edge_path": str(args.edge_path),
        "membership_path": str(membership_path),
        "target_pct": args.target_pct,
        "nano_min_size": args.nano_min_size,
        "cut_min_size": cut_min_size,
        "seed": args.seed,
        "dendrogram_mode": args.dendrogram_mode,
        "n_nodes": len(ctx["gcc_uids"]),
        "n_edges": ctx["edges_gcc"].height,
        "n_contracted_nodes": ctx["n_contracted"],
        "elapsed": round(elapsed, 1),
        "nano": {
            "gamma": ctx["nano"].gamma,
            "n_clusters": ctx["nano"].n_clusters,
            "max_pct": ctx["nano"].max_pct,
            "top5": ctx["nano"].top5,
            "singleton_pct": ctx["nano_qr"].singleton_pct,
        },
        "cut": {
            "feasible": cut.feasible,
            "n_clusters": int((np.bincount(micro_membership) > 0).sum()),
            "max_pct": micro_qr.max_cluster_pct,
            "top5": micro_qr.top5_sizes,
            "singleton_pct": micro_qr.singleton_pct,
            "total_stability": cut.total_stability,
            "cut_nodes": cut.cut_nodes,
            "contracted_cut_clusters": cut.n_clusters,
        },
        "dendrogram": {
            "n_merges": int(len(ctx["linkage"])),
            "height_min": float(ctx["linkage"][-1, 2]) if len(ctx["linkage"]) else 0.0,
            "height_max": float(ctx["linkage"][0, 2]) if len(ctx["linkage"]) else 0.0,
        },
    }
    out_path = args.output / f"{args.field}_hybrid_cut.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    log.info(
        "Cut: %d clusters, max=%.1f%%, feasible=%s",
        payload["cut"]["n_clusters"],
        payload["cut"]["max_pct"],
        cut.feasible,
    )
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
