#!/usr/bin/env python3
"""Compare Emb_full vs bg+nov using combine_edges (proper weights).

Uses combine_edges with CONSENSUS, SUM, NOISY_OR methods instead of
priority_fill (which outputs integer consensus counts).

Usage:
    .venv/bin/python scripts/eval_bgnov_combine.py --field 15 --n-runs 10
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score

from sciscape.linkage.combination import combine_edges
from sciscape.linkage.config import CombineMethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bgnov2")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results_gcc"
MIN_CLUSTER_SIZE = 5


def load_edges(field_id, filename):
    return pl.read_parquet(EDGE_DIR / f"field_{field_id}" / f"{filename}.parquet")


def cosine_to_rank_weight(df: pl.DataFrame) -> pl.DataFrame:
    """Convert cosine similarity to 1/rank per node."""
    fwd = df.select(pl.col("uid1").alias("node"), pl.col("uid2").alias("neighbor"), pl.col("rel_sum2").alias("w"))
    rev = df.select(pl.col("uid2").alias("node"), pl.col("uid1").alias("neighbor"), pl.col("rel_sum2").alias("w"))
    bidi = pl.concat([fwd, rev])
    ranked = bidi.with_columns(pl.col("w").rank("ordinal", descending=True).over("node").alias("rank"))
    ranked = ranked.with_columns((1.0 / pl.col("rank")).alias("rw"))
    return (
        ranked.with_columns(pl.min_horizontal("node", "neighbor").alias("lo"),
                            pl.max_horizontal("node", "neighbor").alias("hi"))
        .group_by("lo", "hi").agg(pl.col("rw").max())
        .rename({"lo": "uid1", "hi": "uid2", "rw": "rel_sum2"})
    )


def build_graph(df):
    all_ids = pl.concat([df["uid1"].alias("id"), df["uid2"].alias("id")]).unique().sort().to_list()
    id2idx = {w: i for i, w in enumerate(all_ids)}
    n = len(all_ids)
    src = df["uid1"].replace_strict(id2idx, return_dtype=pl.Int64).to_numpy()
    dst = df["uid2"].replace_strict(id2idx, return_dtype=pl.Int64).to_numpy()
    weights = df["rel_sum2"].to_numpy().astype(np.float64)
    edges = np.column_stack([src, dst]).tolist()
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights.tolist()
    g = g.simplify(combine_edges="max")
    gcc_idx = g.connected_components().giant().vs.indices
    g_gcc = g.subgraph(gcc_idx)
    gcc_ids = [all_ids[i] for i in gcc_idx]
    return g_gcc, gcc_ids


def auto_gamma(g):
    weights = np.array(g.es["weight"])
    n, m = g.vcount(), g.ecount()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    return max(1e-8, min(float(np.median(weights)) * density * 2.0, 1.0))


def search_gamma(g, target_k, tol=0.1, max_iter=15, seed=0):
    """Fast γ search: sparse probe → bracket → geometric bisection."""
    _cache: dict[float, int] = {}

    def count_k(gamma):
        gamma = round(gamma, 12)  # avoid float noise
        if gamma in _cache:
            return _cache[gamma]
        p = leidenalg.find_partition(g, leidenalg.CPMVertexPartition,
                                     resolution_parameter=gamma, weights="weight", seed=seed)
        k = sum(1 for c in Counter(p.membership).values() if c >= MIN_CLUSTER_SIZE)
        _cache[gamma] = k
        return k

    min_k, max_k = int(target_k * (1 - tol)), int(target_k * (1 + tol))
    ag = auto_gamma(g)
    best_gamma, best_k = ag, count_k(ag)
    best_dist = abs(best_k - target_k)
    log.info("    search_gamma: target=%d±%d, auto_γ=%.2e→k=%d",
             target_k, target_k - min_k, ag, best_k)

    if min_k <= best_k <= max_k:
        return best_gamma, best_k

    # Phase 1: Sparse probe (5 points) to find bracket fast
    probes = sorted(set(
        ag * (10 ** exp) for exp in [-2, -1, 0, 1, 2]
    ))
    probes = [p for p in probes if 1e-10 < p < 10.0]
    results = [(ag, best_k)]

    for gi in probes:
        ki = count_k(gi)
        results.append((gi, ki))
        d = abs(ki - target_k)
        if d < best_dist:
            best_gamma, best_k, best_dist = gi, ki, d
        if min_k <= ki <= max_k:
            log.info("    probe hit: γ=%.2e→k=%d", gi, ki)
            return gi, ki

    # Phase 2: Find bracket from probes
    results.sort(key=lambda x: x[0])
    lo_g, hi_g, lo_k, hi_k = None, None, None, None
    for i in range(len(results) - 1):
        g1, k1 = results[i]
        g2, k2 = results[i + 1]
        if (k1 <= target_k <= k2) or (k2 <= target_k <= k1):
            lo_g, hi_g = g1, g2
            lo_k, hi_k = k1, k2
            break

    if lo_g is None:
        # No bracket found — fill in gaps with 2 more probes
        for exp in [-1.5, 0.5, -0.5, 1.5]:
            gi = ag * (10 ** exp)
            if 1e-10 < gi < 10.0:
                ki = count_k(gi)
                results.append((gi, ki))
                d = abs(ki - target_k)
                if d < best_dist:
                    best_gamma, best_k, best_dist = gi, ki, d
                if min_k <= ki <= max_k:
                    return gi, ki
        results.sort(key=lambda x: x[0])
        for i in range(len(results) - 1):
            g1, k1 = results[i]
            g2, k2 = results[i + 1]
            if (k1 <= target_k <= k2) or (k2 <= target_k <= k1):
                lo_g, hi_g = g1, g2
                lo_k, hi_k = k1, k2
                break

    if lo_g is None:
        log.info("    no bracket found, best: γ=%.2e→k=%d", best_gamma, best_k)
        return best_gamma, best_k

    log.info("    bracket: [%.2e(k=%d), %.2e(k=%d)]", lo_g, lo_k, hi_g, hi_k)

    # Phase 3: Geometric bisection within bracket
    for i in range(max_iter):
        mid = np.sqrt(lo_g * hi_g)
        km = count_k(mid)
        d = abs(km - target_k)
        if d < best_dist:
            best_gamma, best_k, best_dist = mid, km, d
        if min_k <= km <= max_k:
            log.info("    bisect %d: γ=%.2e→k=%d ✓", i, mid, km)
            return mid, km

        # Move bracket
        if (lo_k < target_k and km < target_k) or (lo_k > target_k and km > target_k):
            lo_g, lo_k = mid, km
        else:
            hi_g, hi_k = mid, km

        if abs(hi_g / lo_g - 1) < 1e-4:
            break

    log.info("    converged: γ=%.2e→k=%d (target=%d)", best_gamma, best_k, target_k)
    return best_gamma, best_k


def evaluate(g, gamma, n_runs):
    memberships, qualities, n_clusters = [], [], []
    for seed in range(n_runs):
        part = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma, weights="weight", seed=seed,
        )
        memberships.append(part.membership)
        qualities.append(part.quality())
        n_clusters.append(sum(1 for c in Counter(part.membership).values() if c >= MIN_CLUSTER_SIZE))

    n = g.vcount()
    mem_arr = np.array(memberships)
    edges_arr = np.array(g.get_edgelist())
    s, d = edges_arr[:, 0], edges_arr[:, 1]
    co = np.mean(mem_arr[:, s] == mem_arr[:, d], axis=0)
    ns = np.zeros(n); nd = np.zeros(n)
    np.add.at(ns, s, co); np.add.at(ns, d, co)
    np.add.at(nd, s, 1.0); np.add.at(nd, d, 1.0)
    m = nd > 0; ns[m] /= nd[m]; ns[~m] = 1.0

    sample = min(20, n_runs)
    rng = np.random.RandomState(42)
    idx_s = rng.choice(n_runs, sample, replace=False) if n_runs > sample else np.arange(n_runs)
    nmis = []
    for i in range(len(idx_s)):
        for j in range(i + 1, len(idx_s)):
            nmis.append(normalized_mutual_info_score(memberships[idx_s[i]], memberships[idx_s[j]]))

    return {
        "n_nodes": n,
        "n_edges": g.ecount(),
        "gamma": gamma,
        "n_clusters_mean": float(np.mean(n_clusters)),
        "n_clusters_std": float(np.std(n_clusters)),
        "nmi_mean": float(np.mean(nmis)) if nmis else 0.0,
        "node_stability_mean": float(ns.mean()),
        "stable_pct": float(np.sum(ns > 0.9) / n),
        "unstable_pct": float(np.sum(ns < 0.5) / n),
    }


def run_config(label, layers_dict, method, target_k, n_runs):
    """Combine edges, build graph, find γ for target_k, evaluate."""
    log.info("── %s [%s] ──", label, method.value)
    t0 = time.time()

    if len(layers_dict) == 1:
        df = list(layers_dict.values())[0]
    else:
        df = combine_edges(layers_dict, method=method, pre_normalize=True)

    g, _ = build_graph(df)
    w = np.array(g.es["weight"])
    log.info("  %d nodes, %d edges, w=[%.4f, %.4f], median=%.4f",
             g.vcount(), g.ecount(), w.min(), w.max(), np.median(w))

    if target_k:
        gamma, k_eff = search_gamma(g, target_k)
        log.info("  γ=%.2e → k=%d (target=%d)", gamma, k_eff, target_k)
    else:
        gamma = auto_gamma(g)
        k_eff = None

    result = evaluate(g, gamma, n_runs)
    result["label"] = label
    result["method"] = method.value
    result["k_target"] = target_k
    result["time_sec"] = time.time() - t0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--target-k", type=int, default=None,
                        help="Target cluster count (default: BC auto_gamma k_eff)")
    parser.add_argument("--rank-weight", action="store_true",
                        help="Convert Emb layers to 1/rank before combining")
    args = parser.parse_args()

    field_id = args.field

    emb_full = load_edges(field_id, "emb_full_knn30")
    emb_bg = load_edges(field_id, "emb_bg_knn30")
    emb_nov = load_edges(field_id, "emb_nov_knn30")
    bc = load_edges(field_id, "bc_assoc_strength")
    cc = load_edges(field_id, "cc_assoc_strength")

    if args.rank_weight:
        log.info("Converting Emb layers to 1/rank weights...")
        emb_full = cosine_to_rank_weight(emb_full)
        emb_bg = cosine_to_rank_weight(emb_bg)
        emb_nov = cosine_to_rank_weight(emb_nov)
        log.info("Done: full=%d, bg=%d, nov=%d", emb_full.height, emb_bg.height, emb_nov.height)

    log.info("Loaded: full=%d, bg=%d, nov=%d, BC=%d, CC=%d",
             emb_full.height, emb_bg.height, emb_nov.height, bc.height, cc.height)

    # Get reference k_eff from BC
    if args.target_k is None:
        g_bc, _ = build_graph(bc)
        gamma_bc = auto_gamma(g_bc)
        p = leidenalg.find_partition(g_bc, leidenalg.CPMVertexPartition,
                                     resolution_parameter=gamma_bc, weights="weight", seed=0)
        target_k = sum(1 for c in Counter(p.membership).values() if c >= MIN_CLUSTER_SIZE)
        log.info("BC reference: γ=%.2e → k=%d", gamma_bc, target_k)
    else:
        target_k = args.target_k

    methods = [CombineMethod.CONSENSUS, CombineMethod.SUM, CombineMethod.NOISY_OR]
    results = []

    for method in methods:
        # 1. Emb_full alone (baseline, same for all methods)
        if method == methods[0]:  # only once
            results.append(run_config(
                "Emb_full", {"Emb_full": emb_full}, method,
                target_k=target_k, n_runs=args.n_runs,
            ))

        # 2. bg+nov combined
        results.append(run_config(
            f"bg+nov", {"Emb_bg": emb_bg, "Emb_nov": emb_nov}, method,
            target_k=target_k, n_runs=args.n_runs,
        ))

        # 3. BC + Emb_full
        results.append(run_config(
            f"BC+full", {"BC": bc, "Emb_full": emb_full}, method,
            target_k=target_k, n_runs=args.n_runs,
        ))

        # 4. BC + bg + nov
        results.append(run_config(
            f"BC+bg+nov", {"BC": bc, "Emb_bg": emb_bg, "Emb_nov": emb_nov}, method,
            target_k=target_k, n_runs=args.n_runs,
        ))

    # Print
    print(f"\n{'='*130}")
    print(f"  combine_edges: Emb_full vs bg+nov — Field {field_id} | target_k={target_k}")
    print(f"{'='*130}")
    print(f"{'Label':<16} {'Method':<12} {'Nodes':>8} {'Edges':>10} {'γ':>10} "
          f"{'#Clust':>8} {'NMI':>6} {'NodeStab':>8} {'Stable%':>8} {'Unstbl%':>8}")
    print("-" * 130)
    for r in results:
        print(f"{r['label']:<16} {r.get('method','—'):<12} "
              f"{r['n_nodes']:>8,} {r['n_edges']:>10,} "
              f"{r['gamma']:>10.2e} "
              f"{r['n_clusters_mean']:>7.0f}±{r['n_clusters_std']:<3.0f}"
              f"{r['nmi_mean']:>6.3f} "
              f"{r['node_stability_mean']:>8.3f} "
              f"{r['stable_pct']*100:>7.1f}% "
              f"{r['unstable_pct']*100:>7.1f}%")

    # Save
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_rankw" if args.rank_weight else ""
    with open(out_dir / f"eval_bgnov_combine{suffix}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved")


if __name__ == "__main__":
    main()
