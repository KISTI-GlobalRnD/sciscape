"""E3: Consensus level x cluster-structure analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _common import load_layer_tables, run_combination, save_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: consensus tier analysis")
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    layers = load_layer_tables(args.edge_dir)
    if len(layers) < 2:
        raise ValueError(f"Need at least 2 layers for consensus-tier analysis, found {sorted(layers)}")

    run = run_combination(
        layers,
        strategy="consensus",
        target_pct=args.target_pct,
        top_k=args.top_k,
        min_size=args.min_size,
        n_seeds=args.n_seeds,
    )

    from sciscape.visualization.consensus import (
        compute_consensus_stats,
        compute_consensus_vs_cluster,
        consensus_to_plotly,
        format_consensus_report,
    )

    stats = compute_consensus_stats(layers, top_k=args.top_k)
    cluster_stats = compute_consensus_vs_cluster(
        layers,
        run["membership_map"],
        top_k=args.top_k,
    )
    plotly_json = consensus_to_plotly(stats, cluster_stats)
    report_text = format_consensus_report(stats, cluster_stats)

    for level, row in sorted(cluster_stats.items()):
        log.info(
            "%d-layer: edges=%6d intra=%5.1f%% cross=%5.1f%%",
            level,
            row["n_edges"],
            row["intra_pct"],
            row["cross_pct"],
        )

    stem = args.output / f"{args.field}_consensus_tiers"
    save_json(
        {
            "field": args.field,
            "edge_dir": str(args.edge_dir),
            "target_pct": args.target_pct,
            "top_k": args.top_k,
            "min_size": args.min_size,
            "n_seeds": args.n_seeds,
            "layers": sorted(layers),
            "consensus_stats": stats,
            "cluster_stats": {str(k): v for k, v in cluster_stats.items()},
            "plotly": plotly_json,
        },
        stem.with_suffix(".json"),
    )
    stem.with_suffix(".txt").write_text(report_text + "\n", encoding="utf-8")
    log.info("\nSaved → %s(.json/.txt)", stem)


if __name__ == "__main__":
    main()
