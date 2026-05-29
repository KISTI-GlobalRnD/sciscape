#!/usr/bin/env python3
"""4-layer balanced top-k test.

Uses adaptive per-layer k (matching sparsest layer's avg degree)
to equalize edge contribution before boosted combination.

nohup nice -n 19 python experiments/combination/balanced_4layer.py &
"""
import json, sys, tempfile, time
from collections import Counter
from pathlib import Path
import numpy as np, polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sciscape.linkage.combine import load_and_combine
from sciscape.clustering.leiden_rust import run_leiden_rust, postprocess_small_clusters_rust
from sciscape.clustering.integer_remap import integer_remap

OUT = Path("experiments/combination/results")
edge_dir = Path("workspace/data/linktype_edges_gcc/field_15")
GAMMAS = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("=== 4-layer balanced top-k boosted ===")
combined = load_and_combine(edge_dir,
    ['bc_cosine','cc_cosine','dc_fractional','emb_bg_knn30'],
    strategy='boosted', gcc=True, top_k='balanced')
n = pl.concat([combined['uid1'],combined['uid2']]).n_unique()
combined.write_parquet(OUT / "combined_f15_4L_balanced_boosted.parquet")
log(f"Combined: {n:,} nodes, {combined.height:,} edges")

results = []
with tempfile.TemporaryDirectory() as td:
    remap = integer_remap(combined, Path(td)/'r')
    ie = pl.read_parquet(remap.int_edges_path)
    s = ie['src'].to_numpy().astype(np.uint32)
    d = ie['dst'].to_numpy().astype(np.uint32)
    w = ie['weight'].to_numpy().astype(np.float64)
    log(f"Weight: mean={w.mean():.4f}, max={w.max():.2f}")

    for gamma in GAMMAS:
        t0 = time.perf_counter()
        r = run_leiden_rust(edges_src=s, edges_dst=d, edges_weight=w,
            resolution=gamma, n_nodes=remap.n_nodes, seed=42)
        p = postprocess_small_clusters_rust(resolution=gamma, min_size=100,
            membership=r.membership, edges_src=s, edges_dst=d, edges_weight=w,
            n_nodes=remap.n_nodes, seed=42, gamma_decay=0.5, max_rounds=5,
            use_greedy=True, use_component_merge=True)
        sizes = Counter(p.membership.tolist())
        mx = max(sizes.values())
        top5 = sorted(sizes.values(), reverse=True)[:5]
        elapsed = time.perf_counter() - t0
        log(f"γ={gamma:.0e}: {p.n_clusters} cl, max={mx} ({100*mx/n:.1f}%), top5={top5}, {elapsed:.0f}s")
        results.append({"gamma": gamma, "n_cl": p.n_clusters, "max_pct": round(100*mx/n,1), "top5": top5})

json.dump(results, open(OUT / "balanced_4layer_sweep.json", "w"), indent=2)
log("Done")
