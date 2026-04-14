#!/usr/bin/env python3
"""Field_12 validation + Emb layer addition test.

Batch script — run with nohup:
    nohup nice -n 19 python experiments/combination/field12_and_emb.py &

Tests:
1. Field_12 (citation-poor): top-30 + 1/rank + boosted (BC+CC+DC)
2. Field_15 + Emb: top-30 + 1/rank + boosted (BC+CC+DC+Emb_bg)
   → Does 4-layer boost (4x) improve or hurt?
"""

import json, sys, tempfile, time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sciscape.linkage.combine import load_and_combine
from sciscape.clustering.leiden_rust import run_leiden_rust, postprocess_small_clusters_rust
from sciscape.clustering.integer_remap import integer_remap, join_back_uids

OUT = Path("experiments/combination/results")
OUT.mkdir(parents=True, exist_ok=True)

GAMMAS = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
MIN_SIZE = 100
TOP_K = 30


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_sweep(edge_dir, layers, strategy, label):
    """Run γ sweep, save combined + memberships."""
    log(f"\n{'='*60}")
    log(f"{label}: {layers} strategy={strategy}")
    log(f"{'='*60}")

    # Combined edges
    cache_combined = OUT / f"combined_{label}.parquet"
    if cache_combined.exists():
        combined = pl.read_parquet(cache_combined)
        log(f"Loaded cached: {combined.height:,} edges")
    else:
        combined = load_and_combine(Path(edge_dir), layers,
                                     strategy=strategy, gcc=True, top_k=TOP_K)
        combined.write_parquet(cache_combined)
        log(f"Combined: {combined.height:,} edges, saved")

    n_total = pl.concat([combined["uid1"], combined["uid2"]]).n_unique()
    log(f"Nodes: {n_total:,}")

    results = []
    for gamma in GAMMAS:
        cache_mem = OUT / f"mem_{label}_g{gamma:.0e}.json"
        if cache_mem.exists():
            mem = json.load(open(cache_mem))
            sizes = Counter(mem.values())
            mx = max(sizes.values())
            top5 = sorted(sizes.values(), reverse=True)[:5]
            log(f"  γ={gamma:.0e}: {len(sizes)} cl, max={mx} ({100*mx/n_total:.1f}%) [cached]")
        else:
            with tempfile.TemporaryDirectory() as td:
                remap = integer_remap(combined, Path(td) / "r")
                ie = pl.read_parquet(remap.int_edges_path)
                s = ie["src"].to_numpy().astype(np.uint32)
                d = ie["dst"].to_numpy().astype(np.uint32)
                w = ie["weight"].to_numpy().astype(np.float64)

                t0 = time.perf_counter()
                r = run_leiden_rust(edges_src=s, edges_dst=d, edges_weight=w,
                                     resolution=gamma, n_nodes=remap.n_nodes, seed=42)
                p = postprocess_small_clusters_rust(
                    resolution=gamma, min_size=MIN_SIZE, membership=r.membership,
                    edges_src=s, edges_dst=d, edges_weight=w,
                    n_nodes=remap.n_nodes, seed=42,
                    gamma_decay=0.5, max_rounds=5,
                    use_greedy=True, use_component_merge=True)

                uid_mem = join_back_uids(p.membership.tolist(), remap.node_manifest_path)
                mem = dict(zip(uid_mem["uid"].to_list(),
                               [int(x) for x in uid_mem["cluster"].to_list()]))
                json.dump(mem, open(cache_mem, "w"))

                sizes = Counter(mem.values())
                mx = max(sizes.values())
                top5 = sorted(sizes.values(), reverse=True)[:5]
                elapsed = time.perf_counter() - t0
                log(f"  γ={gamma:.0e}: {len(sizes)} cl, max={mx} ({100*mx/n_total:.1f}%), "
                    f"top5={top5}, {elapsed:.0f}s")

        results.append({
            "label": label, "gamma": gamma,
            "n_clusters": len(sizes), "max_size": mx,
            "max_pct": round(100 * mx / n_total, 1),
            "top5": sorted(sizes.values(), reverse=True)[:5],
        })

    return results


def main():
    all_results = []

    # 1. Field_12: BC+CC+DC boosted
    all_results.extend(run_sweep(
        "data/linktype_edges_gcc/field_12",
        ["bc_cosine", "cc_cosine", "dc_fractional"],
        "boosted", "f12_boosted",
    ))

    # 2. Field_12: BC+CC+DC sum (for comparison)
    all_results.extend(run_sweep(
        "data/linktype_edges_gcc/field_12",
        ["bc_cosine", "cc_cosine", "dc_fractional"],
        "sum", "f12_sum",
    ))

    # 3. Field_15: BC+CC+DC+Emb boosted (4 layers)
    all_results.extend(run_sweep(
        "data/linktype_edges_gcc/field_15",
        ["bc_cosine", "cc_cosine", "dc_fractional", "emb_bg_knn30"],
        "boosted", "f15_4layer_boosted",
    ))

    # 4. Field_15: BC+CC+DC+Emb sum (4 layers, comparison)
    all_results.extend(run_sweep(
        "data/linktype_edges_gcc/field_15",
        ["bc_cosine", "cc_cosine", "dc_fractional", "emb_bg_knn30"],
        "sum", "f15_4layer_sum",
    ))

    # Save all
    json.dump(all_results, open(OUT / "field12_emb_sweep.json", "w"), indent=2, default=str)

    # Print summary
    log(f"\n{'='*80}")
    log(f"{'Label':>22} {'gamma':>8} {'clusters':>8} {'max%':>6} {'top3':>25}")
    log("-" * 80)
    for r in all_results:
        log(f"{r['label']:>22} {r['gamma']:>8.0e} {r['n_clusters']:>8} "
            f"{r['max_pct']:>5.1f}% {str(r['top5'][:3]):>25}")


if __name__ == "__main__":
    main()
