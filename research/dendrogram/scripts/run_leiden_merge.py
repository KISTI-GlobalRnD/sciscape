"""Baseline: Standard Leiden + merge (no hierarchy).

Runs Leiden at the best gamma, then merges small clusters greedily.
This is the standard approach without hierarchical contraction.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sciscape.clustering.auto_gamma import find_gamma
from sciscape.evaluation.stability import evaluate_stability, compute_quality_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Baseline: Leiden+merge")
    parser.add_argument("edge_path", type=Path)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--min-size", type=int, default=30)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    edges = pl.read_parquet(args.edge_path)
    log.info(f"Edges: {edges.height:,}")

    # Find gamma + Leiden + postprocess (standard, no hierarchy)
    result = find_gamma(
        edges,
        target_max_pct=args.target_pct,
        min_size=args.min_size,
        postprocess=True,
    )
    log.info(f"γ={result.gamma:.2e}, {result.n_clusters} clusters, max={result.max_pct}%")

    # Stability
    stab = evaluate_stability(edges, gamma=result.gamma, n_seeds=5,
                              min_size=args.min_size)
    log.info(stab.summary())

    # Quality
    qr = compute_quality_report(edges, result.membership, gamma=result.gamma,
                                target_pct=args.target_pct)
    log.info(qr.summary())

    # Save
    out = {
        "field": args.field,
        "method": "leiden_merge",
        "gamma": result.gamma,
        "n_clusters": result.n_clusters,
        "max_pct": result.max_pct,
        "min_size": args.min_size,
        "top5": result.top5,
        "ami_mean": stab.ami_mean,
        "ami_std": stab.ami_std,
        "ari_mean": stab.ari_mean,
        "singleton_pct": qr.singleton_pct,
    }
    out_path = args.output / f"{args.field}_leiden_merge.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
