#!/usr/bin/env python3
"""γ sweep for sum vs boosted comparison.

Saves combined edges + membership for each (strategy, γ) pair.
Run with nohup for long jobs:
    nohup python experiments/combination/gamma_sweep.py &

Output:
    experiments/combination/results/
        combined_sum.parquet
        combined_boosted.parquet
        mem_sum_g{gamma}.json
        mem_boosted_g{gamma}.json
        sweep_summary.json
"""

import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sciscape.linkage.combine import load_and_combine
from sciscape.clustering.leiden_rust import run_leiden_rust, postprocess_small_clusters_rust
from sciscape.clustering.integer_remap import integer_remap, join_back_uids


EDGE_DIR = Path("data/linktype_edges_gcc/field_15")
LAYERS = ["bc_cosine", "cc_cosine", "dc_fractional"]
OUT = Path("experiments/combination/results")
OUT.mkdir(parents=True, exist_ok=True)

GAMMAS = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
MIN_SIZE = 100
TOP_K = 30


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_combined(strategy):
    """Save combined edge parquet (cached)."""
    path = OUT / f"combined_{strategy}.parquet"
    if path.exists():
        log(f"{strategy}: loading cached {path}")
        return pl.read_parquet(path)
    log(f"{strategy}: combining top-{TOP_K} + 1/rank + {strategy}...")
    t0 = time.perf_counter()
    combined = load_and_combine(EDGE_DIR, LAYERS,
                                strategy=strategy, gcc=True, top_k=TOP_K)
    combined.write_parquet(path)
    n = pl.concat([combined["uid1"], combined["uid2"]]).n_unique()
    log(f"{strategy}: {n:,} nodes, {combined.height:,} edges ({time.perf_counter()-t0:.0f}s)")
    return combined


def run_one(combined, strategy, gamma):
    """Run Leiden + postprocess at one γ, save membership."""
    cache = OUT / f"mem_{strategy}_g{gamma:.0e}.json"
    if cache.exists():
        log(f"  {strategy} γ={gamma:.0e}: cached")
        return json.load(open(cache))

    with tempfile.TemporaryDirectory() as td:
        remap = integer_remap(combined, Path(td) / "r")
        ie = pl.read_parquet(remap.int_edges_path)
        s = ie["src"].to_numpy().astype(np.uint32)
        d = ie["dst"].to_numpy().astype(np.uint32)
        w = ie["weight"].to_numpy().astype(np.float64)

        t0 = time.perf_counter()
        r = run_leiden_rust(
            edges_src=s, edges_dst=d, edges_weight=w,
            resolution=gamma, n_nodes=remap.n_nodes, seed=42, n_iterations=10,
        )
        p = postprocess_small_clusters_rust(
            resolution=gamma, min_size=MIN_SIZE, membership=r.membership,
            edges_src=s, edges_dst=d, edges_weight=w,
            n_nodes=remap.n_nodes, seed=42,
            gamma_decay=0.5, max_rounds=5,
            use_greedy=True, use_component_merge=True,
        )
        uid_mem = join_back_uids(p.membership.tolist(), remap.node_manifest_path)
        mem = dict(zip(
            uid_mem["uid"].to_list(),
            [int(x) for x in uid_mem["cluster"].to_list()],
        ))
        elapsed = time.perf_counter() - t0

    sizes = Counter(mem.values())
    n_total = len(mem)
    mx = max(sizes.values())
    top5 = sorted(sizes.values(), reverse=True)[:5]
    big = sum(1 for v in sizes.values() if v >= MIN_SIZE)

    log(f"  {strategy} γ={gamma:.0e}: {p.n_clusters} cl, max={mx} ({100*mx/n_total:.1f}%), "
        f"big={big}, top5={top5}, {elapsed:.0f}s")

    json.dump(mem, open(cache, "w"))
    return mem


def main():
    log("=== γ sweep: sum vs boosted (top-30 + 1/rank) ===")
    summary = []

    for strategy in ["sum", "boosted"]:
        combined = save_combined(strategy)
        n_total = pl.concat([combined["uid1"], combined["uid2"]]).n_unique()

        for gamma in GAMMAS:
            mem = run_one(combined, strategy, gamma)
            sizes = Counter(mem.values())
            mx = max(sizes.values())
            top5 = sorted(sizes.values(), reverse=True)[:5]
            big = sum(1 for v in sizes.values() if v >= MIN_SIZE)

            summary.append({
                "strategy": strategy,
                "gamma": gamma,
                "n_clusters": len(sizes),
                "n_big": big,
                "max_size": mx,
                "max_pct": round(100 * mx / n_total, 1),
                "top5": top5,
            })

    # Save summary
    json.dump(summary, open(OUT / "sweep_summary.json", "w"), indent=2, default=str)

    log("\n=== SUMMARY ===")
    log(f"{'strategy':>10} {'gamma':>8} {'clusters':>8} {'big':>5} {'max':>7} {'max%':>6} {'top3':>25}")
    log("-" * 80)
    for r in summary:
        log(f"{r['strategy']:>10} {r['gamma']:>8.0e} {r['n_clusters']:>8} {r['n_big']:>5} "
            f"{r['max_size']:>7} {r['max_pct']:>5.1f}% {str(r['top5'][:3]):>25}")

    # Recommend γ: first where max_pct < 3% for both methods
    for gamma in GAMMAS:
        sum_r = next((r for r in summary if r["strategy"] == "sum" and r["gamma"] == gamma), None)
        boost_r = next((r for r in summary if r["strategy"] == "boosted" and r["gamma"] == gamma), None)
        if sum_r and boost_r and sum_r["max_pct"] < 3 and boost_r["max_pct"] < 3:
            log(f"\nRecommended γ={gamma:.0e}: both methods max < 3%")
            break
    else:
        log("\nNo γ found where both methods max < 3%")


if __name__ == "__main__":
    main()
