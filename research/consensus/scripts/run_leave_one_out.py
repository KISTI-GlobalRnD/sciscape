"""E2: Leave-one-out ablation for all-layer consensus clustering."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _common import load_layer_tables, run_combination, save_json, serialize_run

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: leave-one-out ablation")
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    layers = load_layer_tables(args.edge_dir)
    if len(layers) < 3:
        raise ValueError(f"Need at least 3 layers for leave-one-out, found {sorted(layers)}")

    full = run_combination(
        layers,
        strategy="consensus",
        target_pct=args.target_pct,
        top_k=args.top_k,
        min_size=args.min_size,
        n_seeds=args.n_seeds,
    )
    full_result = serialize_run(
        full,
        method="all_consensus",
        strategy="consensus",
        extra={"removed_layer": None, "kind": "full"},
    )

    ablations: list[dict] = []
    log.info("Full model: %s", "+".join(full_result["layers"]))
    log.info(
        "  clusters=%d AMI=%.3f±%.3f",
        full_result["n_clusters"],
        full_result["ami_mean"],
        full_result["ami_std"],
    )

    for removed in sorted(layers):
        subset = {name: table for name, table in layers.items() if name != removed}
        strategy = "consensus" if len(subset) > 1 else "rank"
        run = run_combination(
            subset,
            strategy=strategy,
            target_pct=args.target_pct,
            top_k=args.top_k,
            min_size=args.min_size,
            n_seeds=args.n_seeds,
        )
        result = serialize_run(
            run,
            method=f"drop_{removed}",
            strategy=strategy,
            extra={
                "removed_layer": removed,
                "kind": "leave_one_out",
                "delta_ami": run["stability"].ami_mean - full["stability"].ami_mean,
                "delta_clusters": run["gamma_result"].n_clusters - full["gamma_result"].n_clusters,
                "delta_max_pct": run["gamma_result"].max_pct - full["gamma_result"].max_pct,
            },
        )
        ablations.append(result)
        log.info(
            "Drop %-13s clusters=%4d Δclusters=%+4d AMI=%.3f ΔAMI=%+.3f",
            removed,
            result["n_clusters"],
            result["delta_clusters"],
            result["ami_mean"],
            result["delta_ami"],
        )

    out_path = args.output / f"{args.field}_leave_one_out.json"
    save_json(
        {
            "field": args.field,
            "edge_dir": str(args.edge_dir),
            "target_pct": args.target_pct,
            "top_k": args.top_k,
            "min_size": args.min_size,
            "n_seeds": args.n_seeds,
            "full": full_result,
            "ablations": ablations,
        },
        out_path,
    )
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
