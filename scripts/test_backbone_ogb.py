#!/usr/bin/env python3
"""Backbone comparison on ogbn-arxiv citation network.

169K nodes, 1.17M edges, 40 arXiv subject area labels.
Tests Disparity, LANS, Top-k. Evaluates NMI/ARI vs ground truth.

Since full graph is large, we subsample to 30K nodes via BFS.
"""
from __future__ import annotations
import sys, time, logging
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ogb"); log.setLevel(logging.INFO)


def load_ogbn_arxiv(n_target=30000, seed=42):
    """Load ogbn-arxiv, build undirected weighted graph, subsample."""
    from ogb.nodeproppred import NodePropPredDataset
    import networkx as nx

    log.info("Loading ogbn-arxiv...")
    dataset = NodePropPredDataset(name="ogbn-arxiv", root="/tmp/ogb_data")
    graph, labels = dataset[0]

    # graph: dict with 'edge_index', 'num_nodes', 'node_feat'
    edge_index = graph["edge_index"]  # (2, E)
    n_nodes = graph["num_nodes"]
    ground_truth = labels.flatten().tolist()

    log.info("  ogbn-arxiv: %d nodes, %d directed edges, %d labels",
             n_nodes, edge_index.shape[1], len(set(ground_truth)))

    # Build undirected NetworkX graph (weight = 1 for all edges)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u != v:  # skip self-loops
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1.0
            else:
                G.add_edge(u, v, weight=1.0)

    log.info("  Undirected: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # BFS subsample
    if n_target and G.number_of_nodes() > n_target:
        import random
        random.seed(seed)
        # Start from a high-degree node for better connectivity
        degrees = dict(G.degree())
        start_candidates = sorted(degrees, key=degrees.get, reverse=True)[:100]
        start = random.choice(start_candidates)

        visited = set()
        queue = [start]
        visited.add(start)
        while len(visited) < n_target and queue:
            node = queue.pop(0)
            for nbr in G.neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
                    if len(visited) >= n_target:
                        break

        G_sub = G.subgraph(visited).copy()
        # Relabel to 0..n-1
        old_to_new = {old: new for new, old in enumerate(sorted(G_sub.nodes()))}
        G_sub = nx.relabel_nodes(G_sub, old_to_new)
        gt_sub = [ground_truth[old] for old in sorted(visited)]

        log.info("  Subsample: %d nodes, %d edges", G_sub.number_of_nodes(), G_sub.number_of_edges())
        return G_sub, gt_sub

    return G, ground_truth


def nx2ig(G):
    import igraph as ig
    nodes = sorted(G.nodes())
    nm = {n: i for i, n in enumerate(nodes)}
    edges = [(nm[u], nm[v]) for u, v in G.edges()]
    ws = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = ws
    return g


def leiden(g, gamma=0.01, seed=42):
    import leidenalg
    return leidenalg.find_partition(g, leidenalg.CPMVertexPartition,
                                    resolution_parameter=gamma, weights="weight", seed=seed)


def compute_nmi(membership, ground_truth):
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    n = min(len(membership), len(ground_truth))
    nmi = normalized_mutual_info_score(ground_truth[:n], membership[:n])
    ari = adjusted_rand_score(ground_truth[:n], membership[:n])
    return nmi, ari


def compute_basic(g, part, min_size=5):
    mem = part.membership
    cnts = Counter(mem)
    big = {c: s for c, s in cnts.items() if s >= min_size}
    return len(big), sum(big.values()) / g.vcount() if g.vcount() else 0, max(big.values()) if big else 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=30000)
    args = parser.parse_args()

    G_nx, gt = load_ogbn_arxiv(n_target=args.n_nodes)
    n_edges = G_nx.number_of_edges()
    n_nodes = G_nx.number_of_nodes()

    # ogbn-arxiv has integer weights (citation count), use lower gammas
    gammas = [0.01, 0.05, 0.1, 0.5]

    print(f"\n{'='*100}")
    print(f"ogbn-arxiv: {n_nodes} nodes, {n_edges} edges, {len(set(gt))} ground truth classes")
    print(f"{'='*100}")

    results = []

    # ── Baseline ──
    log.info("Running baseline...")
    g_ig = nx2ig(G_nx)
    for gm in gammas:
        part = leiden(g_ig, gamma=gm)
        nc, cov, mx = compute_basic(g_ig, part)
        nmi, ari = compute_nmi(part.membership, gt)
        results.append(("BASELINE", "—", n_edges, 100.0, gm, nc, cov, mx, nmi, ari))

    # ── Disparity ──
    import netbone
    log.info("Computing disparity...")
    t0 = time.perf_counter()
    bb_df = netbone.disparity(G_nx)
    t_df = time.perf_counter() - t0
    log.info("  disparity: %.1fs", t_df)

    for alpha in [0.01, 0.05, 0.1, 0.2, 0.4]:
        try:
            G_f = netbone.threshold_filter(bb_df, alpha, narrate=False)
            ne = G_f.number_of_edges()
            if ne == 0: continue
            g_bb = nx2ig(G_f)
            for gm in gammas:
                part = leiden(g_bb, gamma=gm)
                nc, cov, mx = compute_basic(g_bb, part)
                nmi, ari = compute_nmi(part.membership, gt)
                results.append(("disparity", f"α={alpha}", ne, ne/n_edges*100, gm, nc, cov, mx, nmi, ari))
        except Exception as e:
            log.warning("  disparity α=%s: %s", alpha, e)

    # ── LANS ──
    log.info("Computing LANS...")
    t0 = time.perf_counter()
    bb_lans = netbone.lans(G_nx)
    t_lans = time.perf_counter() - t0
    log.info("  LANS: %.1fs", t_lans)

    for alpha in [0.01, 0.05, 0.1, 0.2]:
        try:
            G_f = netbone.threshold_filter(bb_lans, alpha, narrate=False)
            ne = G_f.number_of_edges()
            if ne == 0: continue
            g_bb = nx2ig(G_f)
            for gm in gammas:
                part = leiden(g_bb, gamma=gm)
                nc, cov, mx = compute_basic(g_bb, part)
                nmi, ari = compute_nmi(part.membership, gt)
                results.append(("lans", f"α={alpha}", ne, ne/n_edges*100, gm, nc, cov, mx, nmi, ari))
        except Exception as e:
            log.warning("  lans α=%s: %s", alpha, e)

    # ── Top-k ──
    log.info("Computing top-k...")
    import networkx as nx
    t0 = time.perf_counter()
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
        g_bb = nx2ig(G_topk)
        for gm in gammas:
            part = leiden(g_bb, gamma=gm)
            nc, cov, mx = compute_basic(g_bb, part)
            nmi, ari = compute_nmi(part.membership, gt)
            results.append(("top_k", f"k={k}", ne, ne/n_edges*100, gm, nc, cov, mx, nmi, ari))
    t_topk = time.perf_counter() - t0
    log.info("  top_k: %.1fs", t_topk)

    # ── Print ──
    print(f"\n{'Method':<15} {'Param':>8} {'Edges':>7} {'%Kept':>6} {'γ':>5} {'#C':>5} {'Cov%':>5} {'Max':>6} {'NMI':>6} {'ARI':>6}")
    print("-" * 90)
    for r in results:
        method, param, ne, pct, gm, nc, cov, mx, nmi, ari = r
        print(f"{method:<15} {param:>8} {ne:>7} {pct:>5.1f}% {gm:>5.2f} {nc:>5} {cov*100:>4.0f}% {mx:>6} {nmi:>6.3f} {ari:>6.3f}")

    # Best NMI summary
    print(f"\n{'='*60}")
    print("BEST NMI PER METHOD (across all γ)")
    print(f"{'='*60}")
    from collections import defaultdict
    best = defaultdict(float)
    best_info = defaultdict(str)
    for r in results:
        method, param, ne, pct, gm, nc, cov, mx, nmi, ari = r
        key = f"{method}({param})"
        if nmi > best[key]:
            best[key] = nmi
            best_info[key] = f"γ={gm}, NMI={nmi:.3f}, ARI={ari:.3f}, edges={ne}({pct:.0f}%)"

    for key in sorted(best.keys(), key=lambda x: -best[x]):
        print(f"  {key:<25} {best_info[key]}")

    print(f"\nExtraction times: disparity={t_df:.1f}s, LANS={t_lans:.1f}s, top_k={t_topk:.1f}s")


if __name__ == "__main__":
    main()
