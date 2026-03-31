#!/usr/bin/env python3
"""Leiden ensemble stability analysis.

Run Leiden many times with different seeds → build co-membership matrix
→ identify which edges are stable vs unstable across runs.

Key questions:
- Which edges always stay in the same cluster? (stable)
- Which edges sometimes split? (unstable → noise candidates)
- Do unstable edges correlate with weight, degree, or hub status?
"""
from __future__ import annotations
import sys, time, logging
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy import sparse

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ens"); log.setLevel(logging.INFO)

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


def leiden_ensemble(g, gamma, n_runs=100, seed_start=0):
    """Run Leiden n_runs times with different seeds."""
    import leidenalg
    memberships = []
    for i in range(n_runs):
        part = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma, weights="weight",
            seed=seed_start + i,
        )
        memberships.append(part.membership)
    return memberships


def compute_edge_comembership(g, memberships):
    """For each edge, compute fraction of runs where endpoints are co-clustered."""
    n_runs = len(memberships)
    edge_list = g.get_edgelist()
    n_edges = len(edge_list)

    co_rates = np.zeros(n_edges, dtype=np.float64)
    for mem in memberships:
        for idx, (u, v) in enumerate(edge_list):
            if mem[u] == mem[v]:
                co_rates[idx] += 1.0
    co_rates /= n_runs
    return co_rates


def compute_node_stability(memberships):
    """For each node, compute how often its cluster assignment changes."""
    n_nodes = len(memberships[0])
    n_runs = len(memberships)

    # For each node: fraction of run-pairs where it has the same assignment
    # Simpler: entropy of cluster distribution
    stabilities = np.zeros(n_nodes)
    for i in range(n_nodes):
        assigns = [m[i] for m in memberships]
        counts = Counter(assigns)
        # Fraction of time in the most common cluster
        stabilities[i] = max(counts.values()) / n_runs
    return stabilities


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=3000)
    parser.add_argument("--n-runs", type=int, default=100)
    args = parser.parse_args()

    g = load_igraph(n_target=args.n_nodes)
    n = g.vcount()
    n_edges = g.ecount()
    weights = np.array(g.es["weight"])
    degrees = np.array(g.degree())

    gammas = [0.005, 0.01, 0.02]

    for gamma in gammas:
        print(f"\n{'='*80}")
        print(f"γ = {gamma}, {args.n_runs} runs, {n} nodes, {n_edges} edges")
        print(f"{'='*80}")

        # ── Ensemble ──
        log.info("Running Leiden ensemble (γ=%.3f, %d runs)...", gamma, args.n_runs)
        t0 = time.perf_counter()
        memberships = leiden_ensemble(g, gamma, n_runs=args.n_runs)
        t_ens = time.perf_counter() - t0
        log.info("  Ensemble: %.1fs (%.2fs/run)", t_ens, t_ens / args.n_runs)

        # ── Cluster count variation ──
        n_clusters = [len(set(m)) for m in memberships]
        print(f"\nCluster count: mean={np.mean(n_clusters):.1f}, "
              f"std={np.std(n_clusters):.1f}, "
              f"range=[{min(n_clusters)}, {max(n_clusters)}]")

        # ── Edge co-membership ──
        log.info("Computing edge co-membership...")
        co_rates = compute_edge_comembership(g, memberships)

        print(f"\nEdge co-membership rate (fraction of runs where u,v in same cluster):")
        print(f"  Mean: {co_rates.mean():.3f}")
        print(f"  Std:  {co_rates.std():.3f}")
        for thr in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            n_below = np.sum(co_rates <= thr)
            print(f"  co_rate ≤ {thr:.1f}: {n_below:>6} edges ({n_below/n_edges*100:>5.1f}%)")

        # ── Stable vs unstable edges ──
        stable = co_rates >= 0.9    # almost always together
        unstable = co_rates <= 0.1  # almost never together
        mixed = (~stable) & (~unstable)

        print(f"\nEdge categories:")
        print(f"  Stable (≥0.9):   {stable.sum():>6} ({stable.sum()/n_edges*100:.1f}%)")
        print(f"  Mixed (0.1-0.9): {mixed.sum():>6} ({mixed.sum()/n_edges*100:.1f}%)")
        print(f"  Unstable (≤0.1): {unstable.sum():>6} ({unstable.sum()/n_edges*100:.1f}%)")

        # ── Correlation with weight ──
        print(f"\nWeight by category:")
        print(f"  Stable:   mean={weights[stable].mean():.4f}, median={np.median(weights[stable]):.4f}")
        if mixed.sum() > 0:
            print(f"  Mixed:    mean={weights[mixed].mean():.4f}, median={np.median(weights[mixed]):.4f}")
        if unstable.sum() > 0:
            print(f"  Unstable: mean={weights[unstable].mean():.4f}, median={np.median(weights[unstable]):.4f}")

        # Weight-comembership correlation
        corr = np.corrcoef(weights, co_rates)[0, 1]
        print(f"  Pearson correlation (weight vs co_rate): {corr:.4f}")

        # ── Correlation with endpoint degree ──
        edge_list = g.get_edgelist()
        max_deg = np.array([max(degrees[u], degrees[v]) for u, v in edge_list])
        min_deg = np.array([min(degrees[u], degrees[v]) for u, v in edge_list])
        avg_deg = (max_deg + min_deg) / 2

        print(f"\nEndpoint max-degree by category:")
        print(f"  Stable:   mean={max_deg[stable].mean():.1f}")
        if mixed.sum() > 0:
            print(f"  Mixed:    mean={max_deg[mixed].mean():.1f}")
        if unstable.sum() > 0:
            print(f"  Unstable: mean={max_deg[unstable].mean():.1f}")

        corr_deg = np.corrcoef(max_deg, co_rates)[0, 1]
        print(f"  Pearson correlation (max_degree vs co_rate): {corr_deg:.4f}")

        # ── Node stability ──
        node_stab = compute_node_stability(memberships)
        print(f"\nNode stability (fraction in most common cluster):")
        print(f"  Mean: {node_stab.mean():.3f}, Std: {node_stab.std():.3f}")
        print(f"  Min: {node_stab.min():.3f}, Max: {node_stab.max():.3f}")
        for thr in [0.5, 0.7, 0.9, 1.0]:
            n_above = np.sum(node_stab >= thr)
            print(f"  stability ≥ {thr:.1f}: {n_above:>5} nodes ({n_above/n*100:.1f}%)")

        # ── Hub nodes stability ──
        top5_mask = degrees >= np.percentile(degrees, 95)
        bottom50_mask = degrees <= np.median(degrees)
        print(f"\nNode stability by degree:")
        print(f"  Top 5% hubs (deg≥{np.percentile(degrees, 95):.0f}): "
              f"stability={node_stab[top5_mask].mean():.3f}")
        print(f"  Bottom 50% (deg≤{np.median(degrees):.0f}): "
              f"stability={node_stab[bottom50_mask].mean():.3f}")

        # ── Co-membership matrix stats (node pairs) ──
        # Build sparse co-membership count: how many runs each edge's pair is co-clustered
        # Check: are there node pairs NOT connected by edge that are frequently co-clustered?
        # (Would suggest missing edges)
        # Skip this for now — too expensive for large n

        # ── What if we remove unstable edges? ──
        if unstable.sum() > 0:
            import leidenalg, igraph as ig
            # Build graph without unstable edges
            keep_mask = co_rates > 0.1
            kept_edges = [edge_list[i] for i in range(n_edges) if keep_mask[i]]
            kept_weights = weights[keep_mask]

            g_stable = ig.Graph(n=n, edges=kept_edges, directed=False)
            g_stable.es["weight"] = kept_weights.tolist()

            # Get GCC
            gcc = g_stable.connected_components().giant()
            g_gcc = g_stable.subgraph(gcc.vs.indices)

            part_stable = leidenalg.find_partition(
                g_gcc, leidenalg.CPMVertexPartition,
                resolution_parameter=gamma, weights="weight", seed=42,
            )
            part_orig = leidenalg.find_partition(
                g, leidenalg.CPMVertexPartition,
                resolution_parameter=gamma, weights="weight", seed=42,
            )

            cnts_orig = Counter(part_orig.membership)
            cnts_stable = Counter(part_stable.membership)
            big_orig = sum(1 for s in cnts_orig.values() if s >= 5)
            big_stable = sum(1 for s in cnts_stable.values() if s >= 5)

            print(f"\nEffect of removing unstable edges:")
            print(f"  Edges: {n_edges} → {sum(keep_mask)} ({sum(keep_mask)/n_edges*100:.1f}%)")
            print(f"  GCC: {n} → {g_gcc.vcount()} nodes")
            print(f"  Clusters (≥5): {big_orig} → {big_stable}")
            print(f"  Max cluster: {max(cnts_orig.values())} → {max(cnts_stable.values())}")

            # Stability of the stable-edge graph
            mems2 = leiden_ensemble(g_gcc, gamma, n_runs=20)
            nc2 = [len(set(m)) for m in mems2]
            print(f"  Cluster count after removal: mean={np.mean(nc2):.1f}, std={np.std(nc2):.1f}")

        print(f"\nEnsemble time: {t_ens:.1f}s")


if __name__ == "__main__":
    main()
