#!/usr/bin/env python3
"""Fast backbone comparison: skip slow methods (ECM, HSS).

Tests: Disparity, Noise-Corrected, LANS, Marginal Likelihood,
       Doubly Stochastic, Global Threshold, MDL (paninipy).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

KRISS_DIR = Path.home() / "Desktop/Workspace/1.4.2.KRISS"
EDGE_PATH = KRISS_DIR / "Data" / "KRISS_pair_links" / "dc_bc_cc_total_pair.txt"


def load_igraph(edge_path, n_target=None, seed=42):
    import igraph as ig
    import polars as pl

    logger.info("Loading: %s", edge_path)
    df = pl.read_csv(edge_path, separator="\t")
    uid_col1, uid_col2, weight_col = df.columns[0], df.columns[1], df.columns[2]
    all_uids = list(set(df[uid_col1].to_list() + df[uid_col2].to_list()))
    uid_to_idx = {u: i for i, u in enumerate(all_uids)}
    sources = [uid_to_idx[u] for u in df[uid_col1].to_list()]
    targets = [uid_to_idx[u] for u in df[uid_col2].to_list()]
    weights = df[weight_col].to_list()
    g = ig.Graph(n=len(all_uids), edges=list(zip(sources, targets)), directed=False)
    g.es["weight"] = weights
    gcc_ids = g.connected_components().giant().vs.indices
    g = g.subgraph(gcc_ids)
    logger.info("  GCC: %d nodes, %d edges", g.vcount(), g.ecount())

    if n_target and g.vcount() > n_target:
        import random
        random.seed(seed)
        start = random.randint(0, g.vcount() - 1)
        bfs_order = g.bfs(start)[0]
        keep = bfs_order[:n_target]
        g = g.subgraph(keep)
        gcc_ids = g.connected_components().giant().vs.indices
        g = g.subgraph(gcc_ids)
        logger.info("  Subsample GCC: %d nodes, %d edges", g.vcount(), g.ecount())
    return g


def igraph_to_networkx(g_ig):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(g_ig.vcount()))
    for e in g_ig.es:
        G.add_edge(e.source, e.target, weight=e["weight"])
    return G


def networkx_to_igraph(G_nx):
    import igraph as ig
    node_list = sorted(G_nx.nodes())
    node_map = {n: i for i, n in enumerate(node_list)}
    edges = [(node_map[u], node_map[v]) for u, v in G_nx.edges()]
    weights = [G_nx[u][v].get("weight", 1.0) for u, v in G_nx.edges()]
    g = ig.Graph(n=len(node_list), edges=edges, directed=False)
    g.es["weight"] = weights
    return g


def run_leiden(g_ig, gamma=0.01, seed=42):
    import leidenalg
    return leidenalg.find_partition(
        g_ig, leidenalg.CPMVertexPartition,
        resolution_parameter=gamma, weights="weight", seed=seed,
    )


def compute_metrics(g_ig, partition, min_size=5):
    membership = partition.membership
    n = g_ig.vcount()
    counts = Counter(membership)
    large = {c: s for c, s in counts.items() if s >= min_size}
    n_clusters = len(large)
    coverage = sum(large.values()) / n if n > 0 else 0
    sizes = sorted(large.values(), reverse=True)
    max_size = sizes[0] if sizes else 0
    median_size = float(np.median(sizes)) if sizes else 0

    densities = []
    for cid in large:
        members = [i for i, m in enumerate(membership) if m == cid]
        if len(members) < 2:
            continue
        sub = g_ig.subgraph(members)
        edge_sum = sum(sub.es["weight"]) if sub.ecount() > 0 else 0
        n_sub = sub.vcount()
        max_edges = n_sub * (n_sub - 1) / 2
        densities.append(edge_sum / max_edges if max_edges > 0 else 0)

    return {
        "n_clusters": n_clusters,
        "coverage": coverage,
        "max_size": max_size,
        "median_size": median_size,
        "avg_density": float(np.mean(densities)) if densities else 0,
        "modularity": partition.modularity,
    }


def test_backbone_leiden(g_backbone_ig, gammas, label):
    """Run Leiden at multiple γ on a backbone graph."""
    # Get GCC
    comps = g_backbone_ig.connected_components()
    gcc_ids = comps.giant().vs.indices
    g = g_backbone_ig.subgraph(gcc_ids)

    results = {}
    for gamma in gammas:
        try:
            part = run_leiden(g, gamma=gamma)
            results[gamma] = compute_metrics(g, part)
        except Exception as e:
            logger.warning("  %s γ=%.3f FAILED: %s", label, gamma, e)
    return results, g.vcount(), g.ecount()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import netbone

    # ── Load ──
    g_ig = load_igraph(EDGE_PATH, n_target=args.n_nodes, seed=args.seed)
    n_nodes = g_ig.vcount()
    n_edges_orig = g_ig.ecount()
    G_nx = igraph_to_networkx(g_ig)
    gammas = [0.005, 0.01, 0.02, 0.05]

    # ── Baseline ──
    print(f"\n{'='*80}")
    print(f"BASELINE: {n_nodes} nodes, {n_edges_orig} edges")
    print(f"{'='*80}")
    baseline = {}
    for gamma in gammas:
        part = run_leiden(g_ig, gamma=gamma)
        baseline[gamma] = compute_metrics(g_ig, part)

    # ── Backbone methods ──
    # Statistical: threshold_filter with alpha
    stat_configs = [
        ("disparity", lambda G: netbone.disparity(G), [0.01, 0.05, 0.1]),
        ("noise_corrected", lambda G: netbone.noise_corrected(G, approximation=True), [0.01, 0.05, 0.1]),
        ("lans", lambda G: netbone.lans(G), [0.01, 0.05, 0.1]),
        ("marginal_likelihood", lambda G: netbone.marginal_likelihood(G), [0.01, 0.05, 0.1]),
    ]

    # Structural: fraction_filter
    struct_configs = [
        ("doubly_stochastic", lambda G: netbone.doubly_stochastic(G), [0.1, 0.3, 0.5]),
        ("global_threshold", lambda G: netbone.global_threshold(G), [0.1, 0.3, 0.5]),
    ]

    all_results = {}  # (method, param) -> {gamma: metrics}
    timing = {}

    # Run statistical backbones
    for name, func, alphas in stat_configs:
        logger.info("Computing %s ...", name)
        t0 = time.perf_counter()
        try:
            bb = func(G_nx)
            elapsed = time.perf_counter() - t0
            timing[name] = elapsed
            logger.info("  %s computed in %.1fs", name, elapsed)

            for alpha in alphas:
                try:
                    G_f = netbone.threshold_filter(bb, alpha, narrate=False)
                    if G_f.number_of_edges() == 0:
                        continue
                    g_bb = networkx_to_igraph(G_f)
                    label = f"{name}(α={alpha})"
                    results, gcc_n, gcc_e = test_backbone_leiden(g_bb, gammas, label)
                    all_results[(name, alpha)] = {
                        "results": results,
                        "edges": G_f.number_of_edges(),
                        "gcc_nodes": gcc_n,
                        "gcc_edges": gcc_e,
                    }
                except Exception as e:
                    logger.warning("  %s α=%s filter failed: %s", name, alpha, e)
        except Exception as e:
            timing[name] = -1
            logger.warning("  %s FAILED: %s", name, e)

    # Run structural backbones
    for name, func, fracs in struct_configs:
        logger.info("Computing %s ...", name)
        t0 = time.perf_counter()
        try:
            bb = func(G_nx)
            elapsed = time.perf_counter() - t0
            timing[name] = elapsed
            logger.info("  %s computed in %.1fs", name, elapsed)

            for frac in fracs:
                try:
                    G_f = netbone.fraction_filter(bb, frac, narrate=False)
                    if G_f.number_of_edges() == 0:
                        continue
                    g_bb = networkx_to_igraph(G_f)
                    label = f"{name}(f={frac})"
                    results, gcc_n, gcc_e = test_backbone_leiden(g_bb, gammas, label)
                    all_results[(name, frac)] = {
                        "results": results,
                        "edges": G_f.number_of_edges(),
                        "gcc_nodes": gcc_n,
                        "gcc_edges": gcc_e,
                    }
                except Exception as e:
                    logger.warning("  %s f=%s filter failed: %s", name, frac, e)
        except Exception as e:
            timing[name] = -1
            logger.warning("  %s FAILED: %s", name, e)

    # MDL backbone (parameter-free)
    logger.info("Computing MDL backbone ...")
    t0 = time.perf_counter()
    try:
        from paninipy.mdl_backboning import MDL_backboning
        elist = [(u, v, d["weight"]) for u, v, d in G_nx.edges(data=True)]
        bg, bl, cg, cl = MDL_backboning(elist, directed=False)
        elapsed = time.perf_counter() - t0
        timing["mdl"] = elapsed
        logger.info("  MDL done in %.1fs (global=%d, local=%d edges)", elapsed, len(bg), len(bl))

        import networkx as nx
        for mdl_name, mdl_edges in [("mdl_global", bg), ("mdl_local", bl)]:
            if len(mdl_edges) == 0:
                continue
            G_mdl = nx.Graph()
            G_mdl.add_nodes_from(range(n_nodes))
            for u, v, w in mdl_edges:
                G_mdl.add_edge(int(u), int(v), weight=float(w))
            g_bb = networkx_to_igraph(G_mdl)
            results, gcc_n, gcc_e = test_backbone_leiden(g_bb, gammas, mdl_name)
            all_results[(mdl_name, "auto")] = {
                "results": results,
                "edges": G_mdl.number_of_edges(),
                "gcc_nodes": gcc_n,
                "gcc_edges": gcc_e,
            }
    except Exception as e:
        timing["mdl"] = -1
        logger.warning("  MDL FAILED: %s", e)

    # ── Print Summary ──
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Original: {n_nodes} nodes, {n_edges_orig} edges\n")

    # Header
    hdr = f"{'Method':<30} {'Param':>6} {'Edges':>7} {'%Kept':>6}"
    for g in gammas:
        hdr += f" | γ={g} (#C/Cov%/Max)"
    print(hdr)
    print("-" * len(hdr))

    # Baseline row
    row = f"{'BASELINE':<30} {'—':>6} {n_edges_orig:>7} {'100%':>6}"
    for g in gammas:
        m = baseline[g]
        row += f" | {m['n_clusters']:>4}/{m['coverage']*100:>4.0f}%/{m['max_size']:>4}"
    print(row)

    # Backbone rows
    for (name, param), data in sorted(all_results.items(), key=lambda x: -data_edges(x)):
        edges = data["edges"]
        pct = edges / n_edges_orig * 100
        p_str = f"{param}" if isinstance(param, str) else f"{param:.2f}"
        row = f"{name:<30} {p_str:>6} {edges:>7} {pct:>5.1f}%"
        for g in gammas:
            if g in data["results"]:
                m = data["results"][g]
                row += f" | {m['n_clusters']:>4}/{m['coverage']*100:>4.0f}%/{m['max_size']:>4}"
            else:
                row += f" | {'—':>14}"
        print(row)

    # Timing
    print(f"\n{'='*80}")
    print("EXTRACTION TIMES")
    print(f"{'='*80}")
    for name, t in sorted(timing.items()):
        if t < 0:
            print(f"  {name:<25} FAILED")
        else:
            print(f"  {name:<25} {t:.1f}s")

    # Density comparison table
    print(f"\n{'='*80}")
    print("INTERNAL DENSITY COMPARISON (avg across clusters ≥5 nodes)")
    print(f"{'='*80}")
    hdr2 = f"{'Method':<30} {'Param':>6}"
    for g in gammas:
        hdr2 += f" | γ={g:>5}"
    print(hdr2)
    print("-" * len(hdr2))

    row = f"{'BASELINE':<30} {'—':>6}"
    for g in gammas:
        row += f" | {baseline[g]['avg_density']:>7.4f}"
    print(row)

    for (name, param), data in sorted(all_results.items(), key=lambda x: -data_edges(x)):
        p_str = f"{param}" if isinstance(param, str) else f"{param:.2f}"
        row = f"{name:<30} {p_str:>6}"
        for g in gammas:
            if g in data["results"]:
                row += f" | {data['results'][g]['avg_density']:>7.4f}"
            else:
                row += f" | {'—':>7}"
        print(row)


def data_edges(item):
    return item[1].get("edges", 0)


if __name__ == "__main__":
    main()
