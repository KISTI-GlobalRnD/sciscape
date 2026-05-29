#!/usr/bin/env python3
"""Multi-layer combination strategy comparison experiment.

Compares Step1 (normalization) × Step2 (combination) strategies
across two extreme fields (citation-rich vs citation-poor).

Usage:
    python experiments/combination/run_comparison.py \
        --fields 15 12 \
        --edge-dir workspace/data/linktype_edges_gcc \
        --output experiments/combination/results

Steps:
1. For each (field, normalization, combination) triple:
   - Combine edges → GCC → Leiden → postprocess
   - Record: n_clusters, size distribution, timing
2. Sample worst-case nodes from each result
3. (Optional) LLM blind comparison between methods
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── Normalization strategies (Step 1) ────────────────────

def normalize_raw(df: pl.DataFrame) -> pl.DataFrame:
    """No normalization — raw weights."""
    return df

def normalize_rank(df: pl.DataFrame) -> pl.DataFrame:
    """1/rank normalization."""
    return (
        df.sort("rel_sum2", descending=True)
        .with_row_index("_rank")
        .with_columns((1.0 / (pl.col("_rank") + 1).cast(pl.Float64)).alias("rel_sum2"))
        .drop("_rank")
    )

def normalize_quantile(df: pl.DataFrame) -> pl.DataFrame:
    """Quantile normalization — map to [0, 1] by percentile rank."""
    n = df.height
    if n == 0:
        return df
    return (
        df.sort("rel_sum2")
        .with_row_index("_rank")
        .with_columns((pl.col("_rank").cast(pl.Float64) / n).alias("rel_sum2"))
        .drop("_rank")
    )

NORMALIZERS = {
    "raw": normalize_raw,
    "rank": normalize_rank,
    "quantile": normalize_quantile,
}

# ── Combination strategies (Step 2) ─────────────────────

def combine_sum(layers: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Sum weights across layers."""
    parts = [df.select("uid1", "uid2", "rel_sum2") for df in layers.values() if df.height > 0]
    if not parts:
        return pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
    return pl.concat(parts).group_by(["uid1", "uid2"]).agg(pl.col("rel_sum2").sum())

def combine_max(layers: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Max weight across layers."""
    parts = [df.select("uid1", "uid2", "rel_sum2") for df in layers.values() if df.height > 0]
    if not parts:
        return pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
    return pl.concat(parts).group_by(["uid1", "uid2"]).agg(pl.col("rel_sum2").max())

def combine_vote(layers: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Binary vote — weight = number of layers containing this edge."""
    parts = []
    for df in layers.values():
        if df.height > 0:
            parts.append(df.select("uid1", "uid2").with_columns(pl.lit(1.0).alias("rel_sum2")))
    if not parts:
        return pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
    return pl.concat(parts).group_by(["uid1", "uid2"]).agg(pl.col("rel_sum2").sum())

def combine_consensus(layers: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Boosted sum: weight × number of layers containing this edge.

    edge(i,j) = sum_of_weights(i,j) × n_layers_present(i,j)
    Multi-layer consensus gets multiplicative bonus.
    """
    parts_w = []  # weighted
    parts_v = []  # vote (binary)
    for df in layers.values():
        if df.height > 0:
            parts_w.append(df.select("uid1", "uid2", "rel_sum2"))
            parts_v.append(df.select("uid1", "uid2").with_columns(pl.lit(1.0).alias("_vote")))
    if not parts_w:
        return pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})

    # Sum of weights
    w_sum = pl.concat(parts_w).group_by(["uid1", "uid2"]).agg(
        pl.col("rel_sum2").sum()
    )
    # Count of layers
    v_count = pl.concat(parts_v).group_by(["uid1", "uid2"]).agg(
        pl.col("_vote").sum().alias("_n_layers")
    )
    # Multiply: weight × n_layers
    combined = w_sum.join(v_count, on=["uid1", "uid2"], how="left").with_columns(
        (pl.col("rel_sum2") * pl.col("_n_layers")).alias("rel_sum2")
    ).drop("_n_layers")
    return combined


COMBINERS = {
    "sum": combine_sum,
    "max": combine_max,
    "vote": combine_vote,
    "consensus": combine_consensus,
}


# ── Pipeline ─────────────────────────────────────────────

def run_one_config(
    edge_dir: Path,
    field: str,
    layer_names: List[str],
    norm_name: str,
    combine_name: str,
    gamma: float,
    min_size: int,
    max_edges_per_layer: int | None = None,
) -> Dict:
    """Run one (normalization, combination) configuration."""
    import tempfile
    from sciscape.linkage.filters import filter_giant_component
    from sciscape.clustering.leiden_rust import run_leiden_rust, postprocess_small_clusters_rust
    from sciscape.clustering.integer_remap import integer_remap

    normalize = NORMALIZERS[norm_name]
    combine = COMBINERS[combine_name]

    t0 = time.perf_counter()

    # Load & normalize each layer
    layers: Dict[str, pl.DataFrame] = {}
    for name in layer_names:
        path = edge_dir / f"field_{field}" / f"{name}.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        if max_edges_per_layer and df.height > max_edges_per_layer:
            df = df.sort("rel_sum2", descending=True).head(max_edges_per_layer)
        df = normalize(df)
        layers[name] = df

    if not layers:
        return {"error": f"No layers found for field_{field}"}

    # Combine
    combined = combine(layers)
    t_combine = time.perf_counter() - t0

    # GCC
    combined = filter_giant_component(combined)
    n_nodes = pl.concat([combined["uid1"], combined["uid2"]]).n_unique()
    t_gcc = time.perf_counter() - t0

    # Leiden
    with tempfile.TemporaryDirectory() as tmpdir:
        remap = integer_remap(combined, Path(tmpdir) / "remap")
        ie = pl.read_parquet(remap.int_edges_path)
        src = ie["src"].to_numpy().astype(np.uint32)
        dst = ie["dst"].to_numpy().astype(np.uint32)
        w = ie["weight"].to_numpy().astype(np.float64)

        t_leiden_start = time.perf_counter()
        r = run_leiden_rust(
            edges_src=src, edges_dst=dst, edges_weight=w,
            resolution=gamma, n_nodes=remap.n_nodes, seed=42, n_iterations=10,
        )
        t_leiden = time.perf_counter() - t_leiden_start

        sizes = Counter(r.membership.tolist())
        big = sum(1 for s in sizes.values() if s >= min_size)

        t_post_start = time.perf_counter()
        p = postprocess_small_clusters_rust(
            resolution=gamma, min_size=min_size, membership=r.membership,
            edges_src=src, edges_dst=dst, edges_weight=w,
            n_nodes=remap.n_nodes, seed=42,
            gamma_decay=0.5, max_rounds=5,
            use_greedy=True, use_component_merge=True,
        )
        t_post = time.perf_counter() - t_post_start

        psizes = Counter(p.membership.tolist())
        pbig = sum(1 for s in psizes.values() if s >= min_size)
        psmall = sum(1 for s in psizes.values() if s < min_size)
        top5 = sorted(psizes.values(), reverse=True)[:5]

    total = time.perf_counter() - t0

    result = {
        "field": field,
        "layers": layer_names,
        "norm": norm_name,
        "combine": combine_name,
        "gamma": gamma,
        "min_size": min_size,
        "n_nodes": n_nodes,
        "n_edges": combined.height,
        "leiden_clusters": r.n_clusters,
        "leiden_big": big,
        "post_clusters": p.n_clusters,
        "post_big": pbig,
        "post_small": psmall,
        "top5_sizes": top5,
        "time_combine": round(t_combine, 1),
        "time_leiden": round(t_leiden, 1),
        "time_post": round(t_post, 1),
        "time_total": round(total, 1),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-layer combination comparison")
    parser.add_argument("--fields", nargs="+", default=["15", "12"])
    parser.add_argument("--edge-dir", type=Path, default=Path("workspace/data/linktype_edges_gcc"))
    parser.add_argument("--output", type=Path, default=Path("experiments/combination/results"))
    parser.add_argument("--gamma", type=float, default=1e-6)
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--max-edges", type=int, default=500_000,
                        help="Max edges per layer (subsample for speed)")
    parser.add_argument("--layers", type=str, default="bc_cosine,cc_cosine,dc_fractional",
                        help="Layer names to combine")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    layer_names = args.layers.split(",")

    results = []

    for field in args.fields:
        log.info(f"\n{'='*60}")
        log.info(f"Field {field}")
        log.info(f"{'='*60}")

        for norm in NORMALIZERS:
            for comb in COMBINERS:
                config_name = f"{norm}_{comb}"
                log.info(f"  {config_name}...")
                try:
                    r = run_one_config(
                        args.edge_dir, field, layer_names,
                        norm, comb, args.gamma, args.min_size,
                        max_edges_per_layer=args.max_edges,
                    )
                    results.append(r)
                    log.info(
                        f"    → {r['post_clusters']} cl ({r['post_big']} big, "
                        f"{r['post_small']} small), top5={r['top5_sizes']}, "
                        f"{r['time_total']:.1f}s"
                    )
                except Exception as e:
                    log.error(f"    FAILED: {e}")
                    results.append({
                        "field": field, "norm": norm, "combine": comb,
                        "error": str(e),
                    })

    # Save results
    out_path = args.output / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_path}")

    # Print summary table
    log.info(f"\n{'Field':>6} {'Norm':>10} {'Combine':>8} {'Nodes':>8} {'Edges':>10} "
             f"{'Post_cl':>7} {'Big':>4} {'Sm':>4} {'Top3':>20} {'Time':>6}")
    log.info("-" * 100)
    for r in results:
        if "error" in r:
            log.info(f"{r.get('field','?'):>6} {r.get('norm','?'):>10} {r.get('combine','?'):>8} ERROR: {r['error']}")
            continue
        top3 = r["top5_sizes"][:3]
        log.info(
            f"{r['field']:>6} {r['norm']:>10} {r['combine']:>8} "
            f"{r['n_nodes']:>8,} {r['n_edges']:>10,} "
            f"{r['post_clusters']:>7} {r['post_big']:>4} {r['post_small']:>4} "
            f"{str(top3):>20} {r['time_total']:>5.1f}s"
        )


if __name__ == "__main__":
    main()
