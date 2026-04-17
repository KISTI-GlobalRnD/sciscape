"""E1: Single-layer vs multi-layer consensus comparison.

For each field, compare clustering quality:
  - BC-only (top-30)
  - CC-only (top-30)
  - DC-only (top-30)
  - BC+CC+DC consensus (top-30 per layer)
  - BC+CC+DC+Emb consensus (top-30 per layer)

Metrics: n_clusters, max_pct, AMI stability (5 seeds), quality report
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sciscape.linkage.combine import combine_edge_layers
from sciscape.clustering.auto_gamma import find_gamma
from sciscape.evaluation.stability import evaluate_stability, compute_quality_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def run_single_layer(layer_name: str, layer_path: Path, target_pct: float = 3.0):
    """Run clustering on a single layer."""
    df = pl.read_parquet(layer_path)
    combined = combine_edge_layers({layer_name: df}, strategy="rank", gcc=True, top_k="auto")
    log.info(f"  {layer_name}: {combined.height:,} edges")

    result = find_gamma(combined, target_max_pct=target_pct, postprocess=True)
    stab = evaluate_stability(combined, gamma=result.gamma, n_seeds=5, min_size=10, postprocess=True)
    qr = compute_quality_report(combined, result.membership, gamma=result.gamma, target_pct=target_pct)

    return {
        "layer": layer_name,
        "n_edges": combined.height,
        "gamma": result.gamma,
        "n_clusters": result.n_clusters,
        "max_pct": result.max_pct,
        "ami_mean": stab.ami_mean,
        "ami_std": stab.ami_std,
        "ari_mean": stab.ari_mean,
        "singleton_pct": qr.singleton_pct,
    }


def run_multi_layer(layers: dict, strategy: str = "consensus", target_pct: float = 3.0):
    """Run clustering on combined multi-layer edges."""
    combined = combine_edge_layers(layers, strategy=strategy, gcc=True, top_k="auto")
    layer_names = "+".join(sorted(layers.keys()))
    log.info(f"  {layer_names} ({strategy}): {combined.height:,} edges")

    result = find_gamma(combined, target_max_pct=target_pct, postprocess=True)
    stab = evaluate_stability(combined, gamma=result.gamma, n_seeds=5, min_size=10, postprocess=True)
    qr = compute_quality_report(combined, result.membership, gamma=result.gamma,
                                target_pct=target_pct, layer_tables=layers)

    return {
        "layers": layer_names,
        "strategy": strategy,
        "n_edges": combined.height,
        "gamma": result.gamma,
        "n_clusters": result.n_clusters,
        "max_pct": result.max_pct,
        "ami_mean": stab.ami_mean,
        "ami_std": stab.ami_std,
        "ari_mean": stab.ari_mean,
        "singleton_pct": qr.singleton_pct,
        "consensus_edges": qr.consensus_edge_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="E1: Single vs multi-layer comparison")
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("--field", type=str, required=True, help="Field identifier (e.g., field_15)")
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    # Discover available layers
    layer_paths = {}
    for name in ["bc_cosine", "cc_cosine", "dc_fractional", "emb_knn"]:
        p = args.edge_dir / f"{name}.parquet"
        if p.exists():
            layer_paths[name] = p
            log.info(f"Found layer: {name} ({pl.read_parquet(p).height:,} edges)")

    if not layer_paths:
        log.error(f"No edge files found in {args.edge_dir}")
        return

    # Single-layer runs
    log.info("\n=== Single-layer ===")
    for name, path in layer_paths.items():
        r = run_single_layer(name, path, args.target_pct)
        results.append(r)
        log.info(f"  → {r['n_clusters']} cl, AMI={r['ami_mean']:.3f}±{r['ami_std']:.3f}")

    # Multi-layer: citation only (BC+CC+DC)
    citation_layers = {k: pl.read_parquet(v) for k, v in layer_paths.items()
                       if k in ("bc_cosine", "cc_cosine", "dc_fractional")}
    if len(citation_layers) >= 2:
        log.info("\n=== Multi-layer (citation) ===")
        r = run_multi_layer(citation_layers, "consensus", args.target_pct)
        results.append(r)
        log.info(f"  → {r['n_clusters']} cl, AMI={r['ami_mean']:.3f}")

    # Multi-layer: all (BC+CC+DC+Emb)
    all_layers = {k: pl.read_parquet(v) for k, v in layer_paths.items()}
    if len(all_layers) >= 3:
        log.info("\n=== Multi-layer (all) ===")
        r = run_multi_layer(all_layers, "consensus", args.target_pct)
        results.append(r)
        log.info(f"  → {r['n_clusters']} cl, AMI={r['ami_mean']:.3f}")

    # Save results
    out_path = args.output / f"{args.field}_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
