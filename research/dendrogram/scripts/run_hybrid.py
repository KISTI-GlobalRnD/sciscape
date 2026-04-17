"""Run the hybrid hierarchy pipeline and export comparable metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sciscape.clustering.hierarchical import LEVEL_NAMES, build_hierarchy
from sciscape.evaluation.stability import compute_quality_report, evaluate_stability

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _uniform_dict(value: float | int, n_levels: int) -> dict[str, float | int]:
    return {
        LEVEL_NAMES[idx] if idx < len(LEVEL_NAMES) else f"level_{idx}": value
        for idx in range(n_levels)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid CPM hierarchy")
    parser.add_argument("edge_path", type=Path)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--n-levels", type=int, default=4)
    parser.add_argument("--target-pct", type=float, default=None, help="Override target max%% for all levels")
    parser.add_argument("--min-size", type=int, default=None, help="Override min cluster size for all levels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-at-clusters", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    edges = pl.read_parquet(args.edge_path)
    log.info("Edges: %d", edges.height)

    targets = _uniform_dict(args.target_pct, args.n_levels) if args.target_pct is not None else None
    min_sizes = _uniform_dict(args.min_size, args.n_levels) if args.min_size is not None else None
    cache_dir = args.cache_dir or (args.output / f"{args.field}_hybrid_cache")

    result = build_hierarchy(
        edges=edges,
        n_levels=args.n_levels,
        targets=targets,
        min_sizes=min_sizes,
        seed=args.seed,
        stop_at_clusters=args.stop_at_clusters,
        cache_dir=cache_dir,
    )

    quality_by_level: list[dict] = []
    for level in result.levels:
        qr = compute_quality_report(
            edges,
            level.membership,
            gamma=level.gamma,
            target_pct=float(args.target_pct) if args.target_pct is not None else level.max_pct,
        )
        quality_by_level.append(
            {
                "name": level.name,
                "gamma": level.gamma,
                "n_clusters": level.n_clusters,
                "max_pct": level.max_pct,
                "avg_size": level.avg_size,
                "top5": level.top5,
                "elapsed": level.elapsed,
                "singleton_pct": qr.singleton_pct,
            }
        )
        log.info(
            "%-8s clusters=%4d max=%5.1f%% gamma=%.2e",
            level.name,
            level.n_clusters,
            level.max_pct,
            level.gamma,
        )

    stability = None
    if result.levels:
        first = result.levels[0]
        stability = evaluate_stability(
            edges,
            gamma=first.gamma,
            n_seeds=5,
            min_size=args.min_size or 30,
            postprocess=True,
        )
        log.info("\n%s", stability.summary())

    deepest = quality_by_level[-1] if quality_by_level else None
    payload = {
        "field": args.field,
        "method": "hybrid_hierarchy",
        "edge_path": str(args.edge_path),
        "cache_dir": str(cache_dir),
        "n_nodes": result.n_nodes,
        "n_levels": len(result.levels),
        "requested_n_levels": args.n_levels,
        "seed": args.seed,
        "stop_at_clusters": args.stop_at_clusters,
        "target_pct": args.target_pct,
        "min_size": args.min_size,
        "levels": quality_by_level,
        "deepest_level": deepest,
        "stability": (
            {
                "ami_mean": stability.ami_mean,
                "ami_std": stability.ami_std,
                "ari_mean": stability.ari_mean,
                "ari_std": stability.ari_std,
            }
            if stability
            else None
        ),
    }
    out_path = args.output / f"{args.field}_hybrid.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
