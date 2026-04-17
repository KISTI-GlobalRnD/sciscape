"""Run hybrid CPM-critical hierarchy on a dataset.

Usage:
    python run_hybrid.py edges.parquet --field field_15 -o results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sciscape.clustering.hierarchical import build_hierarchy
from sciscape.evaluation.stability import evaluate_stability, compute_quality_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Hybrid CPM hierarchy")
    parser.add_argument("edge_path", type=Path)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--n-levels", type=int, default=4)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    edges = pl.read_parquet(args.edge_path)
    log.info(f"Edges: {edges.height:,}")

    # Run hierarchy
    cache_dir = args.output / f"{args.field}_hybrid_cache"
    result = build_hierarchy(
        edges=edges,
        n_levels=args.n_levels,
        cache_dir=cache_dir,
    )

    # Per-level stats
    level_stats = []
    for level in result.levels:
        log.info(f"{level.name}: {level.n_clusters} cl, max={level.max_pct}%, γ={level.gamma:.2e}")
        level_stats.append({
            "name": level.name,
            "gamma": level.gamma,
            "n_clusters": level.n_clusters,
            "max_pct": level.max_pct,
            "avg_size": level.avg_size,
            "top5": level.top5,
        })

    # Stability on first level
    if result.levels:
        first = result.levels[0]
        stab = evaluate_stability(edges, gamma=first.gamma, n_seeds=5, min_size=30)
        log.info(f"\n{stab.summary()}")

        qr = compute_quality_report(edges, first.membership, gamma=first.gamma)
        log.info(f"\n{qr.summary()}")

    # Save
    out = {
        "field": args.field,
        "n_levels": len(result.levels),
        "n_nodes": result.n_nodes,
        "levels": level_stats,
        "stability": {
            "ami_mean": stab.ami_mean, "ami_std": stab.ami_std,
            "ari_mean": stab.ari_mean,
        } if result.levels else None,
    }
    out_path = args.output / f"{args.field}_hybrid.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
