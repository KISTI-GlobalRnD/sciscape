#!/usr/bin/env python3
"""Inspect individual Leiden run quality.

Questions:
- Are individual runs "good" (dense clusters) even though they differ?
- How different are runs from each other? (pairwise NMI)
- Is the landscape degenerate (many equally good solutions)?
- What do specific clusters look like?
"""
from __future__ import annotations
import sys, time, logging
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("q"); log.setLevel(logging.INFO)

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


def leiden(g, gamma, seed=42):
    import leidenalg
    return leidenalg.find_partition(
        g, leidenalg.CPMVertexPartition,
        resolution_parameter=gamma, weights="weight", seed=seed)


def partition_quality(g, part, min_size=5):
    """Compute detailed quality metrics for a partition."""
    mem = part.membership
    n = g.vcount()
    cnts = Counter(mem)

    all_sizes = sorted(cnts.values(), reverse=True)
    big = {c: s for c, s in cnts.items() if s >= min_size}
    big_sizes = sorted(big.values(), reverse=True)

    # Internal density per cluster
    densities = []
    internal_weights = []
    for cid in big:
        members = [i for i, m in enumerate(mem) if m == cid]
        sub = g.subgraph(members)
        ew = sum(sub.es["weight"]) if sub.ecount() > 0 else 0
        ns = sub.vcount()
        mx = ns * (ns - 1) / 2
        densities.append(ew / mx if mx > 0 else 0)
        internal_weights.append(ew)

    # CPM quality: sum over clusters of (internal_edges - gamma * n_c * (n_c-1)/2)
    cpm_quality = part.quality()

    # Coverage: fraction of total weight inside clusters
    total_weight = sum(g.es["weight"])
    internal_total = sum(internal_weights)

    return {
        "n_clusters_all": len(cnts),
        "n_clusters_big": len(big),
        "singletons": sum(1 for s in cnts.values() if s == 1),
        "coverage_nodes": sum(big.values()) / n,
        "coverage_weight": internal_total / total_weight if total_weight > 0 else 0,
        "max_size": big_sizes[0] if big_sizes else 0,
        "median_size": float(np.median(big_sizes)) if big_sizes else 0,
        "mean_density": float(np.mean(densities)) if densities else 0,
        "min_density": float(np.min(densities)) if densities else 0,
        "cpm_quality": cpm_quality,
        "sizes": big_sizes,
    }


def pairwise_nmi(memberships, sample_n=20):
    """Compute pairwise NMI between sampled runs."""
    from sklearn.metrics import normalized_mutual_info_score
    n = len(memberships)
    idx = np.random.choice(n, size=min(sample_n, n), replace=False)
    idx.sort()

    nmis = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            nmi = normalized_mutual_info_score(
                memberships[idx[i]], memberships[idx[j]])
            nmis.append(nmi)
    return np.array(nmis)


def main():
    g = load_igraph(n_target=3000)
    n = g.vcount()
    gamma = 0.01

    print(f"\n{'='*80}")
    print(f"INDIVIDUAL RUN QUALITY: {n} nodes, {g.ecount()} edges, γ={gamma}")
    print(f"{'='*80}")

    # ── Run 20 individual Leiden runs ──
    n_runs = 100
    log.info("Running %d Leiden runs...", n_runs)

    partitions = []
    memberships = []
    qualities = []

    for seed in range(n_runs):
        part = leiden(g, gamma, seed=seed)
        partitions.append(part)
        memberships.append(part.membership)
        q = partition_quality(g, part)
        qualities.append(q)

    # ── Per-run quality ──
    print(f"\n{'─'*80}")
    print("INDIVIDUAL RUN METRICS (20 sampled runs)")
    print(f"{'─'*80}")
    print(f"{'Seed':>4} {'#C(≥5)':>7} {'Single':>7} {'Cov%':>5} {'Max':>5} "
          f"{'Med':>5} {'AvgDens':>8} {'MinDens':>8} {'CPM_Q':>10}")
    print("-" * 75)

    sample_seeds = [0, 1, 2, 5, 10, 20, 30, 42, 50, 60, 70, 80, 90, 99,
                    3, 7, 13, 25, 55, 77]
    for s in sample_seeds:
        q = qualities[s]
        print(f"{s:>4} {q['n_clusters_big']:>7} {q['singletons']:>7} "
              f"{q['coverage_nodes']*100:>4.0f}% {q['max_size']:>5} "
              f"{q['median_size']:>5.0f} {q['mean_density']:>8.4f} "
              f"{q['min_density']:>8.4f} {q['cpm_quality']:>10.2f}")

    # ── Aggregate statistics across runs ──
    print(f"\n{'─'*80}")
    print("AGGREGATE ACROSS ALL 100 RUNS")
    print(f"{'─'*80}")

    metrics = {
        "n_clusters_big": [q["n_clusters_big"] for q in qualities],
        "singletons": [q["singletons"] for q in qualities],
        "coverage_nodes": [q["coverage_nodes"] for q in qualities],
        "max_size": [q["max_size"] for q in qualities],
        "mean_density": [q["mean_density"] for q in qualities],
        "min_density": [q["min_density"] for q in qualities],
        "cpm_quality": [q["cpm_quality"] for q in qualities],
    }

    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 62)
    for name, vals in metrics.items():
        arr = np.array(vals)
        fmt = ".4f" if "density" in name else ".1f" if "quality" in name else ".1f"
        print(f"{name:<20} {arr.mean():>10{fmt}} {arr.std():>10{fmt}} "
              f"{arr.min():>10{fmt}} {arr.max():>10{fmt}}")

    # ── Pairwise NMI ──
    print(f"\n{'─'*80}")
    print("PAIRWISE NMI BETWEEN RUNS")
    print(f"{'─'*80}")

    nmis = pairwise_nmi(memberships, sample_n=30)
    print(f"Pairwise NMI (30 sampled runs, {len(nmis)} pairs):")
    print(f"  Mean: {nmis.mean():.4f}")
    print(f"  Std:  {nmis.std():.4f}")
    print(f"  Min:  {nmis.min():.4f}")
    print(f"  Max:  {nmis.max():.4f}")

    # Distribution
    for thr in [0.7, 0.8, 0.9, 0.95]:
        frac = np.mean(nmis >= thr)
        print(f"  NMI ≥ {thr}: {frac*100:.1f}% of pairs")

    # ── CPM quality comparison: are all runs equally "good"? ──
    print(f"\n{'─'*80}")
    print("CPM QUALITY (objective function value)")
    print(f"{'─'*80}")

    cpm_vals = np.array([q["cpm_quality"] for q in qualities])
    print(f"  Mean: {cpm_vals.mean():.4f}")
    print(f"  Std:  {cpm_vals.std():.4f}")
    print(f"  CoV:  {cpm_vals.std()/abs(cpm_vals.mean())*100:.2f}%")
    print(f"  Range: [{cpm_vals.min():.4f}, {cpm_vals.max():.4f}]")
    print(f"  Best seed: {np.argmax(cpm_vals)} (Q={cpm_vals.max():.4f})")
    print(f"  Worst seed: {np.argmin(cpm_vals)} (Q={cpm_vals.min():.4f})")

    # ── Size distribution comparison ──
    print(f"\n{'─'*80}")
    print("CLUSTER SIZE DISTRIBUTIONS (3 sample runs)")
    print(f"{'─'*80}")

    for s in [0, 42, 99]:
        q = qualities[s]
        sizes = q["sizes"][:15]  # top 15 clusters
        print(f"  Seed {s:>2}: {sizes}")

    # ── Which nodes are ALWAYS in the same cluster? ──
    print(f"\n{'─'*80}")
    print("CONSENSUS: NODES ALWAYS CO-CLUSTERED")
    print(f"{'─'*80}")

    # Build consensus: for each pair of nodes connected by edge,
    # check if they're in same cluster in ALL runs
    edge_list = g.get_edgelist()
    always_together = 0
    never_together = 0
    sometimes = 0

    for u, v in edge_list:
        same = sum(1 for m in memberships if m[u] == m[v])
        if same == n_runs:
            always_together += 1
        elif same == 0:
            never_together += 1
        else:
            sometimes += 1

    print(f"  Always same cluster:    {always_together:>6} ({always_together/len(edge_list)*100:.1f}%)")
    print(f"  Never same cluster:     {never_together:>6} ({never_together/len(edge_list)*100:.1f}%)")
    print(f"  Sometimes (ambiguous):  {sometimes:>6} ({sometimes/len(edge_list)*100:.1f}%)")

    # ── Consensus partition ──
    # Build co-membership matrix → threshold → find connected components
    print(f"\n{'─'*80}")
    print("CONSENSUS PARTITION (co-membership ≥ 50%)")
    print(f"{'─'*80}")

    import igraph as ig
    # Keep edges where co-membership > 50%
    consensus_edges = []
    consensus_weights = []
    for idx, (u, v) in enumerate(edge_list):
        same = sum(1 for m in memberships if m[u] == m[v])
        rate = same / n_runs
        if rate > 0.5:
            consensus_edges.append((u, v))
            consensus_weights.append(g.es[idx]["weight"])

    g_con = ig.Graph(n=n, edges=consensus_edges, directed=False)
    g_con.es["weight"] = consensus_weights

    # Connected components = consensus clusters
    comps = g_con.connected_components()
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    big_comps = [s for s in comp_sizes if s >= 5]

    print(f"  Consensus edges: {len(consensus_edges)} ({len(consensus_edges)/len(edge_list)*100:.1f}%)")
    print(f"  Components: {len(comps)} total, {len(big_comps)} with ≥5 nodes")
    print(f"  Coverage: {sum(big_comps)/n*100:.1f}% of nodes")
    print(f"  Top sizes: {comp_sizes[:15]}")

    # Compare consensus partition quality
    consensus_mem = comps.membership
    import leidenalg
    # Run Leiden on consensus graph
    gcc = g_con.subgraph(g_con.connected_components().giant().vs.indices)
    if gcc.vcount() > 1 and gcc.ecount() > 0:
        part_con = leiden(gcc, gamma, seed=42)
        q_con = partition_quality(gcc, part_con)
        print(f"\n  Leiden on consensus graph:")
        print(f"    Nodes: {gcc.vcount()}, Edges: {gcc.ecount()}")
        print(f"    Clusters(≥5): {q_con['n_clusters_big']}, "
              f"Coverage: {q_con['coverage_nodes']*100:.0f}%, "
              f"AvgDensity: {q_con['mean_density']:.4f}")


if __name__ == "__main__":
    main()
