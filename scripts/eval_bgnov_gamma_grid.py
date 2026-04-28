#!/usr/bin/env python3
"""Compare Emb_full vs bg+nov across multiple γ values.

Runs both configs at the same γ grid to see if bg+nov ever wins.

Usage:
    .venv/bin/python scripts/eval_bgnov_gamma_grid.py --field 15 --n-runs 10
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
from sklearn.metrics import normalized_mutual_info_score

from sciscape.linkage.combination import priority_fill_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("grid")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results_gcc"
MIN_CLUSTER_SIZE = 5


def load_edges(field_id, filename):
    return pl.read_parquet(EDGE_DIR / f"field_{field_id}" / f"{filename}.parquet")


def build_graph_from_df(df):
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
        "n_clusters_mean": float(np.mean(n_clusters)),
        "n_clusters_std": float(np.std(n_clusters)),
        "nmi_mean": float(np.mean(nmis)) if nmis else 0.0,
        "node_stability_mean": float(ns.mean()),
        "stable_pct": float(np.sum(ns > 0.9) / n),
        "unstable_pct": float(np.sum(ns < 0.5) / n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--k-pool", type=int, default=30)
    args = parser.parse_args()

    field_id = args.field

    emb_full = load_edges(field_id, "emb_full_knn30")
    emb_bg = load_edges(field_id, "emb_bg_knn30")
    emb_nov = load_edges(field_id, "emb_nov_knn30")
    bc = load_edges(field_id, "bc_assoc_strength")

    # Build graphs once
    log.info("Building Emb_full graph...")
    g_full, _ = build_graph_from_df(emb_full)
    log.info("  %d nodes, %d edges", g_full.vcount(), g_full.ecount())

    log.info("Building bg+nov consensus graph...")
    df_bgnov = priority_fill_edges({"Emb_bg": emb_bg, "Emb_nov": emb_nov}, k=args.k, k_pool=args.k_pool)
    g_bgnov, _ = build_graph_from_df(df_bgnov)
    log.info("  %d nodes, %d edges", g_bgnov.vcount(), g_bgnov.ecount())

    log.info("Building BC+Emb_full consensus graph...")
    df_bc_full = priority_fill_edges({"BC": bc, "Emb_full": emb_full}, k=args.k, k_pool=args.k_pool)
    g_bc_full, _ = build_graph_from_df(df_bc_full)
    log.info("  %d nodes, %d edges", g_bc_full.vcount(), g_bc_full.ecount())

    log.info("Building BC+bg+nov consensus graph...")
    df_bc_bgnov = priority_fill_edges({"BC": bc, "Emb_bg": emb_bg, "Emb_nov": emb_nov}, k=args.k, k_pool=args.k_pool)
    g_bc_bgnov, _ = build_graph_from_df(df_bc_bgnov)
    log.info("  %d nodes, %d edges", g_bc_bgnov.vcount(), g_bc_bgnov.ecount())

    # γ grid: from few clusters to many
    # For Emb cosine, k=290 needs γ≈8e-3; for consensus graphs, weight is 1-2
    # Use a grid that covers both regimes
    configs = {
        "Emb_full": g_full,
        "bg+nov": g_bgnov,
        "BC+Emb_full": g_bc_full,
        "BC+bg+nov": g_bc_bgnov,
    }

    # Determine γ grid per config type
    # Emb raw: γ in [5e-4, 1e-2]
    # Consensus: weight is integer {1,2,3}, so γ in [0.1, 2.0]
    gamma_grids = {
        "Emb_full": [5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.5e-2],
        "bg+nov": [0.05, 0.1, 0.2, 0.5, 0.8, 1.2],
        "BC+Emb_full": [0.05, 0.1, 0.2, 0.5, 0.8, 1.2],
        "BC+bg+nov": [0.05, 0.1, 0.2, 0.5, 0.8, 1.2],
    }

    all_results = []
    for label, g in configs.items():
        for gamma in gamma_grids[label]:
            log.info("── %s γ=%.2e ──", label, gamma)
            t0 = time.time()
            r = evaluate(g, gamma, args.n_runs)
            r["label"] = label
            r["gamma"] = gamma
            r["n_nodes"] = g.vcount()
            r["n_edges"] = g.ecount()
            r["time_sec"] = time.time() - t0
            all_results.append(r)
            log.info("  k=%.0f±%.0f NMI=%.3f Stab=%.3f Stable=%.1f%% Unstable=%.1f%%",
                     r["n_clusters_mean"], r["n_clusters_std"],
                     r["nmi_mean"], r["node_stability_mean"],
                     r["stable_pct"]*100, r["unstable_pct"]*100)

    # Print grouped by ~similar cluster count
    print(f"\n{'='*130}")
    print(f"  Emb_full vs bg+nov — γ Grid — Field {field_id}")
    print(f"{'='*130}")
    print(f"{'Label':<20} {'γ':>10} {'#Clust':>10} {'NMI':>6} {'NodeStab':>8} {'Stable%':>8} {'Unstbl%':>8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['label']:<20} {r['gamma']:>10.2e} "
              f"{r['n_clusters_mean']:>8.0f}±{r['n_clusters_std']:<3.0f}"
              f"{r['nmi_mean']:>6.3f} "
              f"{r['node_stability_mean']:>8.3f} "
              f"{r['stable_pct']*100:>7.1f}% "
              f"{r['unstable_pct']*100:>7.1f}%")

    # Save
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_bgnov_gamma_grid.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
