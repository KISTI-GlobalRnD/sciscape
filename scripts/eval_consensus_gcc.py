#!/usr/bin/env python3
"""Evaluate consensus-based link-type combinations on oa26_gcc_only.

Uses priority_fill_edges() — per-node rank + consensus slot-filling.
No global weight normalization needed.

Usage:
    # 5 layers (DC/BC/CC/Emb_bg/Emb_nov)
    .venv/bin/python scripts/eval_consensus_gcc.py --field 15 --n-runs 10

    # 4 layers with Emb_full instead of bg/nov
    .venv/bin/python scripts/eval_consensus_gcc.py --field 15 --n-runs 10 --emb-mode full

    # Specific k
    .venv/bin/python scripts/eval_consensus_gcc.py --field 15 --n-runs 10 --k 30
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_cons")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results_gcc"


def load_edges(field_id: int, filename: str) -> pl.DataFrame:
    path = EDGE_DIR / f"field_{field_id}" / f"{filename}.parquet"
    return pl.read_parquet(path)


def auto_gamma(g: ig.Graph) -> float:
    weights = np.array(g.es["weight"])
    n, m = g.vcount(), g.ecount()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    gamma = float(np.median(weights)) * density * 2.0
    gamma = max(gamma, 1e-6)
    gamma = min(gamma, 0.1)
    return gamma


def leiden_ensemble(g, gamma, n_runs):
    memberships, qualities, n_clusters = [], [], []
    for seed in range(n_runs):
        part = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma, weights="weight", seed=seed,
        )
        memberships.append(part.membership)
        qualities.append(part.quality())
        n_clusters.append(sum(1 for c in Counter(part.membership).values() if c >= 5))
    return memberships, qualities, n_clusters


def compute_stability(memberships, g):
    n = g.vcount()
    n_runs = len(memberships)
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

    degrees = np.array(g.degree())
    top10 = max(1, int(n * 0.1))
    hub_idx = np.argsort(degrees)[-top10:]

    sample = min(20, n_runs)
    idx_s = np.random.RandomState(42).choice(n_runs, sample, replace=False)
    nmis = []
    for i in range(len(idx_s)):
        for j in range(i + 1, len(idx_s)):
            nmis.append(normalized_mutual_info_score(memberships[idx_s[i]], memberships[idx_s[j]]))

    return {
        "node_stability_mean": float(node_stab.mean()),
        "stable_pct": float(np.sum(node_stab > 0.9) / n),
        "unstable_pct": float(np.sum(node_stab < 0.5) / n),
        "hub_stability": float(node_stab[hub_idx].mean()),
        "edge_co_mean": float(co_rates.mean()),
        "nmi_mean": float(np.mean(nmis)),
        "nmi_std": float(np.std(nmis)),
    }


def evaluate_combined(
    combined_edges: pl.DataFrame,
    label: str,
    gamma: float | None = None,
    n_runs: int = 10,
) -> dict:
    """Build graph from combined edges and evaluate."""
    log.info("── Evaluating %s ──", label)
    t0 = time.time()

    # Build igraph
    all_ids = (
        pl.concat([combined_edges["uid1"].alias("id"), combined_edges["uid2"].alias("id")])
        .unique().sort().to_list()
    )
    id2idx = {w: i for i, w in enumerate(all_ids)}
    n = len(all_ids)

    src = combined_edges["uid1"].replace_strict(id2idx, return_dtype=pl.Int64).to_numpy()
    dst = combined_edges["uid2"].replace_strict(id2idx, return_dtype=pl.Int64).to_numpy()
    weights = combined_edges["rel_sum2"].to_numpy().astype(np.float64)

    edges = np.column_stack([src, dst]).tolist()
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights.tolist()
    g = g.simplify(combine_edges="max")

    # GCC
    gcc_idx = g.connected_components().giant().vs.indices
    g_gcc = g.subgraph(gcc_idx)

    log.info("  %s: %d nodes, %d edges → GCC: %d, %d",
             label, n, len(edges), g_gcc.vcount(), g_gcc.ecount())

    gamma_used = gamma or auto_gamma(g_gcc)
    log.info("  γ = %.2e", gamma_used)

    memberships, qualities, n_clusters = leiden_ensemble(g_gcc, gamma_used, n_runs)
    stab = compute_stability(memberships, g_gcc)

    return {
        **stab,
        "gamma": gamma_used,
        "n_nodes": g_gcc.vcount(),
        "n_edges": g_gcc.ecount(),
        "n_clusters_mean": float(np.mean(n_clusters)),
        "n_clusters_std": float(np.std(n_clusters)),
        "time_sec": time.time() - t0,
        "label": label,
    }


def print_results(results: list[dict], field_id: int):
    print(f"\n{'='*110}")
    print(f"  CONSENSUS COMBINATION EVALUATION — Field {field_id}")
    print(f"{'='*110}")
    print(f"{'Label':<35} {'Nodes':>8} {'Edges':>10} {'γ':>10} "
          f"{'#Clust':>8} {'NMI':>6} {'NodeStab':>8} {'HubStab':>8} "
          f"{'Stable%':>8} {'Unstbl%':>8}")
    print("-" * 110)
    for r in results:
        print(f"{r['label']:<35} {r['n_nodes']:>8,} {r['n_edges']:>10,} "
              f"{r['gamma']:>10.2e} "
              f"{r['n_clusters_mean']:>7.0f}±{r['n_clusters_std']:<3.0f}"
              f"{r['nmi_mean']:>6.3f} "
              f"{r['node_stability_mean']:>8.3f} {r['hub_stability']:>8.3f} "
              f"{r['stable_pct']*100:>7.1f}% "
              f"{r['unstable_pct']*100:>7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--k", type=int, default=30,
                        help="Slots per node in combined graph")
    parser.add_argument("--k-pool", type=int, default=30,
                        help="Candidates per node per layer")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--emb-mode", choices=["split", "full", "both"], default="both",
                        help="split=bg+nov, full=emb_full, both=run both configs")
    args = parser.parse_args()

    field_id = args.field
    edge_dir = EDGE_DIR / f"field_{field_id}"

    # Load available layers
    layers_citation = {}
    for label, filename in [("DC", "dc_fractional"), ("BC", "bc_assoc_strength"), ("CC", "cc_assoc_strength")]:
        path = edge_dir / f"{filename}.parquet"
        if path.exists():
            layers_citation[label] = pl.read_parquet(path)
            log.info("Loaded %s: %d edges", label, layers_citation[label].height)

    layers_emb_full = {}
    if (edge_dir / "emb_full_knn30.parquet").exists():
        layers_emb_full["Emb"] = pl.read_parquet(edge_dir / "emb_full_knn30.parquet")
        log.info("Loaded Emb_full: %d edges", layers_emb_full["Emb"].height)

    layers_emb_split = {}
    for label, filename in [("Emb_bg", "emb_bg_knn30"), ("Emb_nov", "emb_nov_knn30")]:
        path = edge_dir / f"{filename}.parquet"
        if path.exists():
            layers_emb_split[label] = pl.read_parquet(path)
            log.info("Loaded %s: %d edges", label, layers_emb_split[label].height)

    results = []

    # Config 1: Citation only (DC+BC+CC)
    if len(layers_citation) >= 2:
        log.info("=== Citation only: %s ===", list(layers_citation.keys()))
        combined = priority_fill_edges(
            layers_citation, k=args.k, k_pool=args.k_pool,
        )
        r = evaluate_combined(combined, "citation_only", gamma=args.gamma, n_runs=args.n_runs)
        results.append(r)

    # Config 2: Citation + Emb_full
    if args.emb_mode in ("full", "both") and layers_emb_full:
        all_layers = {**layers_citation, **layers_emb_full}
        log.info("=== Citation + Emb_full: %s ===", list(all_layers.keys()))
        combined = priority_fill_edges(
            all_layers, k=args.k, k_pool=args.k_pool,
        )
        r = evaluate_combined(combined, "citation+emb_full", gamma=args.gamma, n_runs=args.n_runs)
        results.append(r)

    # Config 3: Citation + Emb_bg + Emb_nov (5 layers)
    if args.emb_mode in ("split", "both") and len(layers_emb_split) == 2:
        all_layers = {**layers_citation, **layers_emb_split}
        log.info("=== Citation + Emb_bg + Emb_nov: %s ===", list(all_layers.keys()))
        combined = priority_fill_edges(
            all_layers, k=args.k, k_pool=args.k_pool,
        )
        r = evaluate_combined(combined, "citation+emb_bg+nov", gamma=args.gamma, n_runs=args.n_runs)
        results.append(r)

    # Config 4: Emb only (for comparison)
    if layers_emb_full:
        log.info("=== Emb_full only ===")
        combined = priority_fill_edges(
            layers_emb_full, k=args.k, k_pool=args.k_pool,
        )
        r = evaluate_combined(combined, "emb_full_only", gamma=args.gamma, n_runs=args.n_runs)
        results.append(r)

    # Print
    print_results(results, field_id)

    # Save
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_consensus.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
