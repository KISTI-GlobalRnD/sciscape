#!/usr/bin/env python3
"""Compare Emb_full alone vs bg+nov combined via priority_fill_edges.

Key question: does splitting into bg/nov and recombining beat using full?

Configs tested:
  1. Emb_full alone
  2. priority_fill(Emb_bg, Emb_nov)  — bg+nov consensus
  3. BC + Emb_full
  4. BC + Emb_bg + Emb_nov
  5. BC + CC + Emb_full
  6. BC + CC + Emb_bg + Emb_nov

Usage:
    .venv/bin/python scripts/eval_bgnov_vs_full.py --field 15 --n-runs 10
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

from sciscape.linkage.combination import priority_fill_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bgnov")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results_gcc"
MIN_CLUSTER_SIZE = 5


def load_edges(field_id, filename):
    return pl.read_parquet(EDGE_DIR / f"field_{field_id}" / f"{filename}.parquet")


def build_graph_from_df(df):
    """Build igraph GCC from edge DataFrame."""
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
    gamma = float(np.median(weights)) * density * 2.0
    return max(1e-8, min(gamma, 1.0))


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
    src, dst = edges_arr[:, 0], edges_arr[:, 1]
    co_rates = np.mean(mem_arr[:, src] == mem_arr[:, dst], axis=0)

    node_stab = np.zeros(n)
    node_deg = np.zeros(n)
    np.add.at(node_stab, src, co_rates)
    np.add.at(node_stab, dst, co_rates)
    np.add.at(node_deg, src, 1.0)
    np.add.at(node_deg, dst, 1.0)
    mask = node_deg > 0
    node_stab[mask] /= node_deg[mask]
    node_stab[~mask] = 1.0

    sample = min(20, n_runs)
    rng = np.random.RandomState(42)
    idx_s = rng.choice(n_runs, sample, replace=False) if n_runs > sample else np.arange(n_runs)
    nmis = []
    for i in range(len(idx_s)):
        for j in range(i + 1, len(idx_s)):
            nmis.append(normalized_mutual_info_score(memberships[idx_s[i]], memberships[idx_s[j]]))

    return {
        "n_nodes": g.vcount(),
        "n_edges": g.ecount(),
        "gamma": gamma,
        "n_clusters_mean": float(np.mean(n_clusters)),
        "n_clusters_std": float(np.std(n_clusters)),
        "nmi_mean": float(np.mean(nmis)) if nmis else 0.0,
        "node_stability_mean": float(node_stab.mean()),
        "stable_pct": float(np.sum(node_stab > 0.9) / n),
        "unstable_pct": float(np.sum(node_stab < 0.5) / n),
    }


def run_config(label, layers_dict, k, k_pool, gamma, n_runs):
    """Run priority_fill_edges then evaluate."""
    log.info("── %s ──", label)
    t0 = time.time()

    if len(layers_dict) == 1:
        # Single layer: use directly
        df = list(layers_dict.values())[0]
        g, node_ids = build_graph_from_df(df)
    else:
        combined = priority_fill_edges(layers_dict, k=k, k_pool=k_pool)
        g, node_ids = build_graph_from_df(combined)

    gamma_used = gamma or auto_gamma(g)
    log.info("  %s: %d nodes, %d edges, γ=%.2e", label, g.vcount(), g.ecount(), gamma_used)

    result = evaluate(g, gamma_used, n_runs)
    result["label"] = label
    result["time_sec"] = time.time() - t0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--k-pool", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=None)
    args = parser.parse_args()

    field_id = args.field

    # Load all layers
    emb_full = load_edges(field_id, "emb_full_knn30")
    emb_bg = load_edges(field_id, "emb_bg_knn30")
    emb_nov = load_edges(field_id, "emb_nov_knn30")
    bc = load_edges(field_id, "bc_assoc_strength")
    cc = load_edges(field_id, "cc_assoc_strength")

    log.info("Loaded: Emb_full=%d, Emb_bg=%d, Emb_nov=%d, BC=%d, CC=%d",
             emb_full.height, emb_bg.height, emb_nov.height, bc.height, cc.height)

    results = []

    # ── Embedding only ──
    # 1. Emb_full alone
    results.append(run_config(
        "Emb_full_alone", {"Emb_full": emb_full},
        k=args.k, k_pool=args.k_pool, gamma=args.gamma, n_runs=args.n_runs,
    ))

    # 2. bg+nov consensus
    results.append(run_config(
        "bg+nov_consensus", {"Emb_bg": emb_bg, "Emb_nov": emb_nov},
        k=args.k, k_pool=args.k_pool, gamma=args.gamma, n_runs=args.n_runs,
    ))

    # ── With BC ──
    # 3. BC + Emb_full
    results.append(run_config(
        "BC+Emb_full", {"BC": bc, "Emb_full": emb_full},
        k=args.k, k_pool=args.k_pool, gamma=args.gamma, n_runs=args.n_runs,
    ))

    # 4. BC + Emb_bg + Emb_nov
    results.append(run_config(
        "BC+bg+nov", {"BC": bc, "Emb_bg": emb_bg, "Emb_nov": emb_nov},
        k=args.k, k_pool=args.k_pool, gamma=args.gamma, n_runs=args.n_runs,
    ))

    # ── With BC+CC ──
    # 5. BC + CC + Emb_full
    results.append(run_config(
        "BC+CC+Emb_full", {"BC": bc, "CC": cc, "Emb_full": emb_full},
        k=args.k, k_pool=args.k_pool, gamma=args.gamma, n_runs=args.n_runs,
    ))

    # 6. BC + CC + Emb_bg + Emb_nov
    results.append(run_config(
        "BC+CC+bg+nov", {"BC": bc, "CC": cc, "Emb_bg": emb_bg, "Emb_nov": emb_nov},
        k=args.k, k_pool=args.k_pool, gamma=args.gamma, n_runs=args.n_runs,
    ))

    # Print
    print(f"\n{'='*120}")
    print(f"  Emb_full vs bg+nov — Field {field_id} | priority_fill k={args.k}")
    print(f"{'='*120}")
    print(f"{'Label':<25} {'Nodes':>8} {'Edges':>10} {'γ':>10} "
          f"{'#Clust':>8} {'NMI':>6} {'NodeStab':>8} {'Stable%':>8} {'Unstbl%':>8}")
    print("-" * 120)
    for r in results:
        print(f"{r['label']:<25} {r['n_nodes']:>8,} {r['n_edges']:>10,} "
              f"{r['gamma']:>10.2e} "
              f"{r['n_clusters_mean']:>7.0f}±{r['n_clusters_std']:<3.0f}"
              f"{r['nmi_mean']:>6.3f} "
              f"{r['node_stability_mean']:>8.3f} "
              f"{r['stable_pct']*100:>7.1f}% "
              f"{r['unstable_pct']*100:>7.1f}%")

    # Save
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_bgnov_vs_full.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
