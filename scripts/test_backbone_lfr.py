#!/usr/bin/env python3
"""Backbone comparison on LFR synthetic benchmark with ground truth.

Tests Disparity, LANS, Top-k on LFR at different mixing parameters (mu).
Evaluates NMI/ARI against planted communities.
"""
from __future__ import annotations
import sys, time, logging
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lfr"); log.setLevel(logging.INFO)


def generate_lfr(n=10000, mu=0.3, seed=42):
    """Generate weighted LFR benchmark graph."""
    import networkx as nx
    log.info("Generating LFR: n=%d, mu=%.2f", n, mu)
    G = nx.LFR_benchmark_graph(
        n=n, tau1=2.5, tau2=1.5, mu=mu,
        average_degree=20, max_degree=100,
        min_community=20, max_community=500,
        seed=seed,
    )
    # Extract ground truth communities
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    # Assign membership (first community for overlapping)
    node_to_comm = {}
    for cid, comm in enumerate(communities):
        for v in comm:
            if v not in node_to_comm:
                node_to_comm[v] = cid

    # Add weights (LFR generates unweighted by default in networkx)
    # Use community structure: intra-community edges get weight 1.0,
    # inter-community edges get weight mu
    for u, v in G.edges():
        if node_to_comm.get(u) == node_to_comm.get(v):
            G[u][v]["weight"] = 1.0
        else:
            G[u][v]["weight"] = mu

    ground_truth = [node_to_comm.get(i, -1) for i in range(n)]
    log.info("  LFR: %d nodes, %d edges, %d communities",
             G.number_of_nodes(), G.number_of_edges(), len(communities))
    return G, ground_truth


def nx2ig(G):
    import igraph as ig
    nodes = sorted(G.nodes())
    nm = {n: i for i, n in enumerate(nodes)}
    edges = [(nm[u], nm[v]) for u, v in G.edges()]
    ws = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = ws
    return g, nm


def leiden(g, gamma=0.01, seed=42):
    import leidenalg
    return leidenalg.find_partition(g, leidenalg.CPMVertexPartition,
                                    resolution_parameter=gamma, weights="weight", seed=seed)


def compute_nmi_ari(membership, ground_truth, node_map=None):
    """Compute NMI and ARI between detected and ground truth."""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    # Align membership to ground truth ordering
    if node_map:
        pred = [membership[node_map[i]] for i in range(len(ground_truth))]
    else:
        pred = membership[:len(ground_truth)]
    nmi = normalized_mutual_info_score(ground_truth, pred)
    ari = adjusted_rand_score(ground_truth, pred)
    return nmi, ari


def compute_basic_metrics(g, part, min_size=5):
    mem = part.membership
    cnts = Counter(mem)
    big = {c: s for c, s in cnts.items() if s >= min_size}
    return {
        "nc": len(big),
        "cov": sum(big.values()) / g.vcount() if g.vcount() else 0,
        "max": max(big.values()) if big else 0,
    }


def run_backbone_test(G_nx, ground_truth, gammas, n_edges_orig):
    """Run all backbone methods and evaluate."""
    import netbone

    results = []

    # ── Baseline ──
    g_ig, nm = nx2ig(G_nx)
    for gm in gammas:
        part = leiden(g_ig, gamma=gm)
        m = compute_basic_metrics(g_ig, part)
        nmi, ari = compute_nmi_ari(part.membership, ground_truth)
        results.append(("BASELINE", "—", n_edges_orig, 100.0, gm, m["nc"], m["cov"], m["max"], nmi, ari))

    # ── Disparity ──
    log.info("  Computing disparity...")
    bb_df = netbone.disparity(G_nx)
    for alpha in [0.01, 0.05, 0.1, 0.2]:
        try:
            G_f = netbone.threshold_filter(bb_df, alpha, narrate=False)
            ne = G_f.number_of_edges()
            if ne == 0: continue
            g_bb, nm_bb = nx2ig(G_f)
            gcc = g_bb.connected_components().giant()
            g_gcc = g_bb.subgraph(gcc.vs.indices)
            for gm in gammas:
                part = leiden(g_gcc, gamma=gm)
                m = compute_basic_metrics(g_gcc, part)
                # For NMI: need to map back
                # Simplify: use full graph partition
                part_full = leiden(g_bb, gamma=gm)
                nmi, ari = compute_nmi_ari(part_full.membership, ground_truth)
                pct = ne / n_edges_orig * 100
                results.append((f"disparity", f"α={alpha}", ne, pct, gm, m["nc"], m["cov"], m["max"], nmi, ari))
        except Exception as e:
            log.warning("  disparity α=%s: %s", alpha, e)

    # ── LANS ──
    log.info("  Computing LANS...")
    bb_lans = netbone.lans(G_nx)
    for alpha in [0.01, 0.05, 0.1, 0.2]:
        try:
            G_f = netbone.threshold_filter(bb_lans, alpha, narrate=False)
            ne = G_f.number_of_edges()
            if ne == 0: continue
            g_bb, nm_bb = nx2ig(G_f)
            gcc = g_bb.connected_components().giant()
            g_gcc = g_bb.subgraph(gcc.vs.indices)
            for gm in gammas:
                part = leiden(g_gcc, gamma=gm)
                m = compute_basic_metrics(g_gcc, part)
                part_full = leiden(g_bb, gamma=gm)
                nmi, ari = compute_nmi_ari(part_full.membership, ground_truth)
                pct = ne / n_edges_orig * 100
                results.append((f"lans", f"α={alpha}", ne, pct, gm, m["nc"], m["cov"], m["max"], nmi, ari))
        except Exception as e:
            log.warning("  lans α=%s: %s", alpha, e)

    # ── Top-k ──
    log.info("  Computing top-k...")
    import networkx as nx
    for k in [10, 20, 30, 50]:
        G_topk = nx.Graph()
        G_topk.add_nodes_from(G_nx.nodes())
        for node in G_nx.nodes():
            nbrs = [(n, G_nx[node][n]["weight"]) for n in G_nx.neighbors(node)]
            nbrs.sort(key=lambda x: -x[1])
            for n, w in nbrs[:k]:
                if G_topk.has_edge(node, n):
                    G_topk[node][n]["weight"] = max(G_topk[node][n]["weight"], w)
                else:
                    G_topk.add_edge(node, n, weight=w)
        ne = G_topk.number_of_edges()
        g_bb, nm_bb = nx2ig(G_topk)
        gcc = g_bb.connected_components().giant()
        g_gcc = g_bb.subgraph(gcc.vs.indices)
        for gm in gammas:
            part = leiden(g_gcc, gamma=gm)
            m = compute_basic_metrics(g_gcc, part)
            part_full = leiden(g_bb, gamma=gm)
            nmi, ari = compute_nmi_ari(part_full.membership, ground_truth)
            pct = ne / n_edges_orig * 100
            results.append((f"top_k", f"k={k}", ne, pct, gm, m["nc"], m["cov"], m["max"], nmi, ari))

    return results


def main():
    gammas = [0.1, 0.3, 0.5, 0.8]  # LFR needs higher γ (intra-weight=1.0)

    all_results = []

    for mu in [0.1, 0.3, 0.5]:
        print(f"\n{'='*100}")
        print(f"LFR BENCHMARK: n=10000, mu={mu}")
        print(f"{'='*100}")

        G, gt = generate_lfr(n=10000, mu=mu, seed=42)
        n_edges = G.number_of_edges()

        results = run_backbone_test(G, gt, gammas, n_edges)

        # Print per-mu table
        print(f"\n{'Method':<15} {'Param':>8} {'Edges':>7} {'%Kept':>6} {'γ':>5} {'#C':>4} {'Cov%':>5} {'Max':>5} {'NMI':>6} {'ARI':>6}")
        print("-" * 85)
        for r in results:
            method, param, ne, pct, gm, nc, cov, mx, nmi, ari = r
            print(f"{method:<15} {param:>8} {ne:>7} {pct:>5.1f}% {gm:>5.1f} {nc:>4} {cov*100:>4.0f}% {mx:>5} {nmi:>6.3f} {ari:>6.3f}")

        all_results.extend([(mu, *r) for r in results])

    # ── Best NMI per method per mu ──
    print(f"\n{'='*100}")
    print("BEST NMI PER METHOD (across all γ)")
    print(f"{'='*100}")
    print(f"{'Method':<15} {'Param':>8} {'mu=0.1':>8} {'mu=0.3':>8} {'mu=0.5':>8}")
    print("-" * 50)

    from collections import defaultdict
    best = defaultdict(lambda: defaultdict(lambda: 0.0))
    best_label = defaultdict(lambda: defaultdict(str))

    for mu, method, param, ne, pct, gm, nc, cov, mx, nmi, ari in all_results:
        key = f"{method}({param})"
        if nmi > best[key][mu]:
            best[key][mu] = nmi
            best_label[key][mu] = f"{nmi:.3f}"

    for key in sorted(best.keys()):
        parts = key.split("(")
        method = parts[0]
        param = parts[1].rstrip(")")
        print(f"{method:<15} {param:>8} {best[key].get(0.1, 0):>8.3f} {best[key].get(0.3, 0):>8.3f} {best[key].get(0.5, 0):>8.3f}")


if __name__ == "__main__":
    main()
