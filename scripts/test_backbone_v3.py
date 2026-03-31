#!/usr/bin/env python3
"""Backbone comparison v3: only proven-fast methods.

Disparity, Noise-Corrected, LANS, Global Threshold, MDL.
Skip: ECM (O(n²) JAX), Doubly Stochastic (Sinkhorn slow), Marginal Likelihood (needs simple graph).
"""
from __future__ import annotations
import sys, time, logging
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bb"); log.setLevel(logging.INFO)

EDGE_PATH = Path.home() / "Desktop/Workspace/1.4.2.KRISS/Data/KRISS_pair_links/dc_bc_cc_total_pair.txt"

def load_igraph(n_target=10000, seed=42):
    import igraph as ig, polars as pl
    log.info("Loading edges...")
    df = pl.read_csv(EDGE_PATH, separator="\t")
    c1, c2, cw = df.columns[0], df.columns[1], df.columns[2]
    uids = list(set(df[c1].to_list() + df[c2].to_list()))
    uid2i = {u: i for i, u in enumerate(uids)}
    g = ig.Graph(n=len(uids), edges=list(zip(
        [uid2i[u] for u in df[c1].to_list()],
        [uid2i[u] for u in df[c2].to_list()]
    )), directed=False)
    g.es["weight"] = df[cw].to_list()
    # Simplify: remove self-loops and merge multi-edges
    g = g.simplify(combine_edges="sum")
    g = g.subgraph(g.connected_components().giant().vs.indices)
    log.info("  Full GCC: %d nodes, %d edges", g.vcount(), g.ecount())
    if n_target and g.vcount() > n_target:
        import random; random.seed(seed)
        start = random.randint(0, g.vcount() - 1)
        keep = g.bfs(start)[0][:n_target]
        g = g.subgraph(keep)
        g = g.simplify(combine_edges="sum")
        g = g.subgraph(g.connected_components().giant().vs.indices)
        log.info("  Subsample GCC: %d nodes, %d edges", g.vcount(), g.ecount())
    return g

def ig2nx(g):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(g.vcount()))
    for e in g.es:
        G.add_edge(e.source, e.target, weight=e["weight"])
    return G

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

def metrics(g, part, min_size=5):
    mem = part.membership; n = g.vcount()
    cnts = Counter(mem)
    big = {c: s for c, s in cnts.items() if s >= min_size}
    sizes = sorted(big.values(), reverse=True)
    dens = []
    for cid in big:
        ms = [i for i, m in enumerate(mem) if m == cid]
        if len(ms) < 2: continue
        sub = g.subgraph(ms)
        ew = sum(sub.es["weight"]) if sub.ecount() > 0 else 0
        mx = sub.vcount() * (sub.vcount() - 1) / 2
        dens.append(ew / mx if mx > 0 else 0)
    return {
        "nc": len(big), "cov": sum(big.values()) / n if n else 0,
        "max": sizes[0] if sizes else 0, "med": float(np.median(sizes)) if sizes else 0,
        "dens": float(np.mean(dens)) if dens else 0, "mod": part.modularity,
    }

def run_on_backbone(g_bb, gammas):
    """Leiden on GCC of backbone graph."""
    g = g_bb.subgraph(g_bb.connected_components().giant().vs.indices)
    res = {}
    for gm in gammas:
        try:
            res[gm] = metrics(g, leiden(g, gamma=gm))
        except Exception as e:
            log.warning("  γ=%.3f failed: %s", gm, e)
    return res, g.vcount(), g.ecount()

def print_table(all_data, gammas, n_orig_edges):
    print(f"\n{'='*110}")
    print(f"{'Method':<32} {'Edges':>7} {'%Kept':>6}", end="")
    for gm in gammas:
        print(f"  |  γ={gm} (#C/Cov%/Max/Dens)", end="")
    print()
    print("-" * 110)

    for label, info in all_data:
        e = info["edges"]; pct = e / n_orig_edges * 100
        print(f"{label:<32} {e:>7} {pct:>5.1f}%", end="")
        for gm in gammas:
            r = info["results"].get(gm)
            if r:
                print(f"  | {r['nc']:>4}/{r['cov']*100:>4.0f}%/{r['max']:>4}/{r['dens']:.3f}", end="")
            else:
                print(f"  | {'—':>22}", end="")
        print()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=10000)
    args = parser.parse_args()

    g_ig = load_igraph(n_target=args.n_nodes)
    n_nodes = g_ig.vcount(); n_edges = g_ig.ecount()
    G_nx = ig2nx(g_ig)
    gammas = [0.005, 0.01, 0.02, 0.05]

    all_data = []
    timing = {}

    # ── Baseline ──
    log.info("Running baseline...")
    res_base = {}
    for gm in gammas:
        res_base[gm] = metrics(g_ig, leiden(g_ig, gamma=gm))
    all_data.append(("BASELINE", {"edges": n_edges, "results": res_base}))

    # ── 1. Disparity Filter ──
    import netbone
    for name, func in [
        ("disparity", lambda: netbone.disparity(G_nx)),
        ("noise_corrected", lambda: netbone.noise_corrected(G_nx, approximation=True)),
        ("lans", lambda: netbone.lans(G_nx)),
    ]:
        log.info("Computing %s...", name)
        t0 = time.perf_counter()
        try:
            bb = func()
            timing[name] = time.perf_counter() - t0
            log.info("  %s: %.1fs", name, timing[name])
            for alpha in [0.01, 0.05, 0.1]:
                try:
                    G_f = netbone.threshold_filter(bb, alpha, narrate=False)
                    ne = G_f.number_of_edges()
                    if ne == 0: continue
                    g_bb = nx2ig(G_f)
                    res, gcc_n, gcc_e = run_on_backbone(g_bb, gammas)
                    label = f"{name}(α={alpha})"
                    all_data.append((label, {"edges": ne, "results": res}))
                except Exception as e:
                    log.warning("  %s α=%s: %s", name, alpha, e)
        except Exception as e:
            timing[name] = -1
            log.warning("  %s FAILED: %s", name, e)

    # ── 2. Global Threshold (fraction-based) ──
    log.info("Computing global_threshold...")
    t0 = time.perf_counter()
    try:
        bb = netbone.global_threshold(G_nx)
        timing["global_threshold"] = time.perf_counter() - t0
        log.info("  global_threshold: %.1fs", timing["global_threshold"])
        for frac in [0.1, 0.3, 0.5]:
            try:
                G_f = netbone.fraction_filter(bb, frac, narrate=False)
                ne = G_f.number_of_edges()
                if ne == 0: continue
                g_bb = nx2ig(G_f)
                res, gcc_n, gcc_e = run_on_backbone(g_bb, gammas)
                all_data.append((f"global_threshold(f={frac})", {"edges": ne, "results": res}))
            except Exception as e:
                log.warning("  global_threshold f=%s: %s", frac, e)
    except Exception as e:
        timing["global_threshold"] = -1
        log.warning("  global_threshold FAILED: %s", e)

    # ── 3. MDL Backbone (parameter-free) ──
    log.info("Computing MDL backbone...")
    t0 = time.perf_counter()
    try:
        from paninipy.mdl_backboning import MDL_backboning
        elist = [(u, v, d["weight"]) for u, v, d in G_nx.edges(data=True)]
        bg, bl, cg, cl = MDL_backboning(elist, directed=False)
        timing["mdl"] = time.perf_counter() - t0
        log.info("  MDL: %.1fs (global=%d, local=%d)", timing["mdl"], len(bg), len(bl))

        import networkx as nx
        for mdl_name, mdl_edges in [("mdl_global", bg), ("mdl_local", bl)]:
            if len(mdl_edges) == 0: continue
            G_m = nx.Graph(); G_m.add_nodes_from(range(n_nodes))
            for u, v, w in mdl_edges:
                G_m.add_edge(int(u), int(v), weight=float(w))
            ne = G_m.number_of_edges()
            g_bb = nx2ig(G_m)
            res, gcc_n, gcc_e = run_on_backbone(g_bb, gammas)
            all_data.append((f"{mdl_name}(auto)", {"edges": ne, "results": res}))
    except Exception as e:
        timing["mdl"] = -1
        log.warning("  MDL FAILED: %s", e)

    # ── 4. Simple top-k per node (our own implementation) ──
    log.info("Computing top-k backbone...")
    t0 = time.perf_counter()
    for k in [10, 20, 30, 50]:
        import networkx as nx
        G_topk = nx.Graph()
        G_topk.add_nodes_from(G_nx.nodes())
        for node in G_nx.nodes():
            neighbors = [(nbr, G_nx[node][nbr]["weight"]) for nbr in G_nx.neighbors(node)]
            neighbors.sort(key=lambda x: -x[1])
            for nbr, w in neighbors[:k]:
                if G_topk.has_edge(node, nbr):
                    # keep max weight
                    G_topk[node][nbr]["weight"] = max(G_topk[node][nbr]["weight"], w)
                else:
                    G_topk.add_edge(node, nbr, weight=w)
        ne = G_topk.number_of_edges()
        g_bb = nx2ig(G_topk)
        res, gcc_n, gcc_e = run_on_backbone(g_bb, gammas)
        all_data.append((f"top_k(k={k})", {"edges": ne, "results": res}))
    timing["top_k"] = time.perf_counter() - t0
    log.info("  top_k: %.1fs", timing["top_k"])

    # ── Print ──
    print_table(all_data, gammas, n_edges)

    print(f"\n{'='*60}")
    print("EXTRACTION TIMES")
    print(f"{'='*60}")
    for name, t in sorted(timing.items()):
        print(f"  {name:<25} {'FAILED' if t < 0 else f'{t:.1f}s'}")

    print(f"\nOriginal graph: {n_nodes} nodes, {n_edges} edges")

if __name__ == "__main__":
    main()
