#!/usr/bin/env python3
"""Compare backbone methods vs ensemble stability.

Question: Does top-k backbone remove the same edges that ensemble identifies
as unstable? If so, top-k is a cheap proxy for ensemble stability.
"""
from __future__ import annotations
import sys, time, logging
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("o"); log.setLevel(logging.INFO)

EDGE_PATH = Path.home() / "Desktop/Workspace/1.4.2.KRISS/Data/KRISS_pair_links/dc_bc_cc_total_pair.txt"


def load_igraph(n_target=3000, seed=42):
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
    g = g.simplify(combine_edges="sum")
    g = g.subgraph(g.connected_components().giant().vs.indices)
    if n_target and g.vcount() > n_target:
        import random; random.seed(seed)
        start = random.randint(0, g.vcount() - 1)
        keep = g.bfs(start)[0][:n_target]
        g = g.subgraph(keep)
        g = g.simplify(combine_edges="sum")
        g = g.subgraph(g.connected_components().giant().vs.indices)
    log.info("  Graph: %d nodes, %d edges", g.vcount(), g.ecount())
    return g


def ig2nx(g):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(g.vcount()))
    for e in g.es:
        G.add_edge(e.source, e.target, weight=e["weight"])
    return G


def main():
    g = load_igraph(n_target=3000)
    n = g.vcount()
    n_edges = g.ecount()
    edge_list = g.get_edgelist()
    edge_set = {(min(u, v), max(u, v)) for u, v in edge_list}
    weights = np.array(g.es["weight"])
    degrees = np.array(g.degree())
    gamma = 0.01

    G_nx = ig2nx(g)

    # ═══════════════════════════════════════════════════
    # 1. ENSEMBLE: compute co-membership per edge
    # ═══════════════════════════════════════════════════
    log.info("Running Leiden ensemble (100 runs)...")
    import leidenalg
    n_runs = 100
    co_rates = np.zeros(n_edges)
    for seed in range(n_runs):
        part = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma, weights="weight", seed=seed)
        mem = part.membership
        for idx, (u, v) in enumerate(edge_list):
            if mem[u] == mem[v]:
                co_rates[idx] += 1
    co_rates /= n_runs

    ensemble_unstable = set(i for i in range(n_edges) if co_rates[i] <= 0.1)
    ensemble_stable = set(i for i in range(n_edges) if co_rates[i] >= 0.9)
    log.info("  Ensemble: %d stable, %d unstable", len(ensemble_stable), len(ensemble_unstable))

    # ═══════════════════════════════════════════════════
    # 2. TOP-K: identify removed edges
    # ═══════════════════════════════════════════════════
    import networkx as nx
    topk_kept = {}
    for k in [10, 20, 30]:
        kept_edges = set()
        for node in G_nx.nodes():
            nbrs = [(n, G_nx[node][n]["weight"]) for n in G_nx.neighbors(node)]
            nbrs.sort(key=lambda x: -x[1])
            for n, w in nbrs[:k]:
                kept_edges.add((min(node, n), max(node, n)))
        # Map back to edge indices
        kept_idx = set()
        for idx, (u, v) in enumerate(edge_list):
            key = (min(u, v), max(u, v))
            if key in kept_edges:
                kept_idx.add(idx)
        removed_idx = set(range(n_edges)) - kept_idx
        topk_kept[k] = kept_idx
        log.info("  Top-k=%d: %d kept, %d removed", k, len(kept_idx), len(removed_idx))

    # ═══════════════════════════════════════════════════
    # 3. DISPARITY + LANS: identify removed edges
    # ═══════════════════════════════════════════════════
    import netbone

    log.info("Computing disparity...")
    bb_df = netbone.disparity(G_nx)
    disp_kept = {}
    for alpha in [0.05, 0.1, 0.2]:
        try:
            G_f = netbone.threshold_filter(bb_df, alpha, narrate=False)
            f_edges = {(min(u, v), max(u, v)) for u, v in G_f.edges()}
            kept_idx = set()
            for idx, (u, v) in enumerate(edge_list):
                if (min(u, v), max(u, v)) in f_edges:
                    kept_idx.add(idx)
            disp_kept[(alpha)] = kept_idx
            log.info("  Disparity α=%s: %d kept", alpha, len(kept_idx))
        except:
            pass

    log.info("Computing LANS...")
    bb_lans = netbone.lans(G_nx)
    lans_kept = {}
    for alpha in [0.05, 0.1, 0.2]:
        try:
            G_f = netbone.threshold_filter(bb_lans, alpha, narrate=False)
            f_edges = {(min(u, v), max(u, v)) for u, v in G_f.edges()}
            kept_idx = set()
            for idx, (u, v) in enumerate(edge_list):
                if (min(u, v), max(u, v)) in f_edges:
                    kept_idx.add(idx)
            lans_kept[(alpha)] = kept_idx
            log.info("  LANS α=%s: %d kept", alpha, len(kept_idx))
        except:
            pass

    # ═══════════════════════════════════════════════════
    # 4. COMPARE: overlap between methods
    # ═══════════════════════════════════════════════════
    all_idx = set(range(n_edges))

    def overlap_stats(removed_by_method, ensemble_unstable, ensemble_stable, label):
        """How well does method's removal align with ensemble instability?"""
        kept_by_method = all_idx - removed_by_method

        # Of edges removed by method, what fraction are ensemble-unstable?
        if len(removed_by_method) > 0:
            precision = len(removed_by_method & ensemble_unstable) / len(removed_by_method)
        else:
            precision = 0

        # Of ensemble-unstable edges, what fraction are removed by method?
        if len(ensemble_unstable) > 0:
            recall = len(removed_by_method & ensemble_unstable) / len(ensemble_unstable)
        else:
            recall = 0

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Jaccard
        union = removed_by_method | ensemble_unstable
        jaccard = len(removed_by_method & ensemble_unstable) / len(union) if len(union) > 0 else 0

        # False positives: removed by method but actually stable
        fp = len(removed_by_method & ensemble_stable)
        fp_rate = fp / len(removed_by_method) if len(removed_by_method) > 0 else 0

        # Mean co-membership of removed vs kept edges
        removed_list = list(removed_by_method)
        kept_list = list(kept_by_method)
        mean_co_removed = co_rates[removed_list].mean() if removed_list else 0
        mean_co_kept = co_rates[kept_list].mean() if kept_list else 0

        return {
            "label": label,
            "removed": len(removed_by_method),
            "pct_removed": len(removed_by_method) / n_edges * 100,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
            "fp_stable": fp,
            "fp_rate": fp_rate,
            "mean_co_removed": mean_co_removed,
            "mean_co_kept": mean_co_kept,
        }

    results = []

    for k, kept in topk_kept.items():
        removed = all_idx - kept
        results.append(overlap_stats(removed, ensemble_unstable, ensemble_stable, f"top_k(k={k})"))

    for alpha, kept in disp_kept.items():
        removed = all_idx - kept
        results.append(overlap_stats(removed, ensemble_unstable, ensemble_stable, f"disparity(α={alpha})"))

    for alpha, kept in lans_kept.items():
        removed = all_idx - kept
        results.append(overlap_stats(removed, ensemble_unstable, ensemble_stable, f"lans(α={alpha})"))

    # ═══════════════════════════════════════════════════
    # 5. PRINT
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"BACKBONE vs ENSEMBLE STABILITY OVERLAP (γ={gamma}, {n} nodes, {n_edges} edges)")
    print(f"Ensemble: {len(ensemble_unstable)} unstable (≤0.1), {len(ensemble_stable)} stable (≥0.9)")
    print(f"{'='*100}")

    print(f"\n{'Method':<22} {'Removed':>7} {'%Rem':>5} "
          f"{'Prec':>6} {'Recall':>6} {'F1':>6} {'Jacc':>6} "
          f"{'FP(stbl)':>8} {'FP%':>5} "
          f"{'CoMem_rem':>10} {'CoMem_kept':>10}")
    print("-" * 105)

    for r in results:
        print(f"{r['label']:<22} {r['removed']:>7} {r['pct_removed']:>4.1f}% "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['jaccard']:>6.3f} "
              f"{r['fp_stable']:>8} {r['fp_rate']:>4.1f}% "
              f"{r['mean_co_removed']:>10.3f} {r['mean_co_kept']:>10.3f}")

    # ═══════════════════════════════════════════════════
    # 6. CO-MEMBERSHIP DISTRIBUTION BY METHOD
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("CO-MEMBERSHIP DISTRIBUTION OF REMOVED EDGES")
    print(f"{'='*100}")

    print(f"\n{'Method':<22}", end="")
    bins = ["0.0", "0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
    for b in bins:
        print(f" {b:>8}", end="")
    print()
    print("-" * 85)

    for label, kept in [
        *[(f"top_k(k={k})", v) for k, v in topk_kept.items()],
        *[(f"disparity(α={a})", v) for a, v in disp_kept.items()],
        *[(f"lans(α={a})", v) for a, v in lans_kept.items()],
    ]:
        removed = list(all_idx - kept)
        if len(removed) == 0:
            continue
        cr = co_rates[removed]
        counts = [
            np.sum(cr == 0),
            np.sum((cr > 0) & (cr <= 0.1)),
            np.sum((cr > 0.1) & (cr <= 0.3)),
            np.sum((cr > 0.3) & (cr <= 0.5)),
            np.sum((cr > 0.5) & (cr <= 0.7)),
            np.sum((cr > 0.7) & (cr <= 0.9)),
            np.sum(cr > 0.9),
        ]
        total = len(removed)
        print(f"{label:<22}", end="")
        for c in counts:
            print(f" {c/total*100:>7.1f}%", end="")
        print()

    # ═══════════════════════════════════════════════════
    # 7. WEIGHT-BASED SIMPLE FILTER vs ENSEMBLE
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("SIMPLE WEIGHT THRESHOLD vs ENSEMBLE")
    print(f"{'='*100}")

    for pct in [10, 20, 30, 40, 50]:
        threshold = np.percentile(weights, pct)
        removed = set(i for i in range(n_edges) if weights[i] <= threshold)
        r = overlap_stats(removed, ensemble_unstable, ensemble_stable, f"weight_bottom_{pct}%")
        print(f"{r['label']:<22} removed={r['removed']:>5} "
              f"prec={r['precision']:.3f} recall={r['recall']:.3f} "
              f"f1={r['f1']:.3f} fp_rate={r['fp_rate']:.1f}%")


if __name__ == "__main__":
    main()
