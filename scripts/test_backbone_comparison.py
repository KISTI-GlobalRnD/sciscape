#!/usr/bin/env python3
"""Compare backbone extraction methods on KRISS 10K subsample.

Tests: Disparity, Noise-Corrected, ECM, LANS, Marginal Likelihood,
       Doubly Stochastic, Global Threshold, MDL (paninipy).

For each backbone: Leiden clustering at multiple γ → cluster count,
internal density, coverage, modularity.

Usage:
    python scripts/test_backbone_comparison.py [--n-nodes 10000]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KRISS_DIR = Path.home() / "Desktop/Workspace/1.4.2.KRISS"
EDGE_PATH = KRISS_DIR / "Data" / "KRISS_pair_links" / "dc_bc_cc_total_pair.txt"


# ── Data Loading ────────────────────────────────────────────────────
def load_igraph(edge_path: Path, n_target: int | None = None, seed: int = 42):
    """Load edge table → igraph GCC → optional BFS subsample."""
    import igraph as ig
    import polars as pl

    logger.info("Loading: %s", edge_path)
    df = pl.read_csv(edge_path, separator="\t")
    logger.info("  %d edges loaded", len(df))

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
    """Convert igraph weighted graph → NetworkX graph."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(g_ig.vcount()))
    for e in g_ig.es:
        G.add_edge(e.source, e.target, weight=e["weight"])
    return G


def networkx_to_igraph(G_nx):
    """Convert NetworkX weighted graph → igraph graph."""
    import igraph as ig

    g = ig.Graph(n=G_nx.number_of_nodes(), directed=False)
    edges = []
    weights = []
    # Map nx node IDs to 0..n-1
    node_map = {n: i for i, n in enumerate(sorted(G_nx.nodes()))}
    for u, v, d in G_nx.edges(data=True):
        edges.append((node_map[u], node_map[v]))
        weights.append(d.get("weight", 1.0))
    g.add_edges(edges)
    g.es["weight"] = weights
    return g


# ── Backbone Extraction ────────────────────────────────────────────
def extract_backbones(G_nx):
    """Run all backbone methods, return dict of {name: backbone_object}."""
    import netbone

    results = {}
    methods = {
        "disparity": lambda G: netbone.disparity(G),
        "noise_corrected": lambda G: netbone.noise_corrected(G, approximation=True),
        "ecm": lambda G: netbone.ecm(G),
        "lans": lambda G: netbone.lans(G),
        "marginal_likelihood": lambda G: netbone.marginal_likelihood(G),
        "doubly_stochastic": lambda G: netbone.doubly_stochastic(G),
        "global_threshold": lambda G: netbone.global_threshold(G),
    }

    for name, func in methods.items():
        logger.info("  Computing backbone: %s ...", name)
        t0 = time.perf_counter()
        try:
            bb = func(G_nx)
            elapsed = time.perf_counter() - t0
            results[name] = {"backbone": bb, "time": elapsed}
            logger.info("    %s done in %.1fs", name, elapsed)
        except Exception as e:
            logger.warning("    %s FAILED: %s", name, e)
            results[name] = {"backbone": None, "time": 0, "error": str(e)}

    return results


def extract_mdl_backbone(G_nx):
    """Run paninipy MDL backbone."""
    from paninipy.mdl_backboning import MDL_backboning

    logger.info("  Computing backbone: MDL ...")
    t0 = time.perf_counter()

    elist = [(u, v, d["weight"]) for u, v, d in G_nx.edges(data=True)]
    bg, bl, cg, cl = MDL_backboning(elist, directed=False)
    elapsed = time.perf_counter() - t0
    logger.info("    MDL done in %.1fs (global: %d edges, local: %d edges)",
                elapsed, len(bg), len(bl))
    return {
        "global": bg,
        "local": bl,
        "compression_global": cg,
        "compression_local": cl,
        "time": elapsed,
    }


def filter_backbone(bb_result, alpha=0.05):
    """Apply threshold filter to a netbone backbone object."""
    import netbone

    bb = bb_result["backbone"]
    if bb is None:
        return None

    try:
        G_filtered = netbone.threshold_filter(bb, alpha, narrate=False)
        return G_filtered
    except Exception:
        try:
            G_filtered = netbone.fraction_filter(bb, alpha, narrate=False)
            return G_filtered
        except Exception:
            try:
                from netbone.filters import boolean_filter
                G_filtered = boolean_filter(bb, narrate=False)
                return G_filtered
            except Exception:
                return None


def mdl_edges_to_networkx(edge_list, n_nodes):
    """Convert MDL backbone edge list to NetworkX graph."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for u, v, w in edge_list:
        G.add_edge(int(u), int(v), weight=float(w))
    return G


# ── Clustering ──────────────────────────────────────────────────────
def run_leiden(g_ig, gamma=0.01, seed=42):
    """Run Leiden CPM on igraph graph, return partition."""
    import leidenalg

    part = leidenalg.find_partition(
        g_ig,
        leidenalg.CPMVertexPartition,
        resolution_parameter=gamma,
        weights="weight",
        seed=seed,
    )
    return part


def compute_metrics(g_ig, partition, min_size=5):
    """Compute clustering quality metrics."""
    membership = partition.membership
    n = g_ig.vcount()
    from collections import Counter
    counts = Counter(membership)

    # Clusters with min_size
    large = {c: s for c, s in counts.items() if s >= min_size}
    n_clusters = len(large)
    coverage = sum(large.values()) / n if n > 0 else 0

    # Sizes
    sizes = sorted(large.values(), reverse=True)
    max_size = sizes[0] if sizes else 0
    median_size = float(np.median(sizes)) if sizes else 0

    # Internal density of large clusters
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

    avg_density = float(np.mean(densities)) if densities else 0

    return {
        "n_clusters": n_clusters,
        "coverage": coverage,
        "max_size": max_size,
        "median_size": median_size,
        "avg_internal_density": avg_density,
        "modularity": partition.modularity,
    }


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1) Load data
    g_ig = load_igraph(EDGE_PATH, n_target=args.n_nodes, seed=args.seed)
    n_nodes = g_ig.vcount()
    n_edges = g_ig.ecount()
    logger.info("Graph ready: %d nodes, %d edges", n_nodes, n_edges)

    # 2) Convert to NetworkX for netbone
    logger.info("Converting to NetworkX...")
    G_nx = igraph_to_networkx(g_ig)
    logger.info("NetworkX graph: %d nodes, %d edges", G_nx.number_of_nodes(), G_nx.number_of_edges())

    # 3) Run baseline (original graph, no backbone)
    gammas = [0.005, 0.01, 0.02, 0.05]
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE (no backbone): %d edges", n_edges)
    logger.info("=" * 70)

    baseline_results = {}
    for gamma in gammas:
        part = run_leiden(g_ig, gamma=gamma)
        metrics = compute_metrics(g_ig, part)
        baseline_results[gamma] = metrics
        logger.info("  γ=%.3f: %d clusters, coverage=%.1f%%, max=%d, density=%.4f",
                     gamma, metrics["n_clusters"], metrics["coverage"] * 100,
                     metrics["max_size"], metrics["avg_internal_density"])

    # 4) Compute all backbones
    logger.info("\n" + "=" * 70)
    logger.info("COMPUTING BACKBONES")
    logger.info("=" * 70)

    bb_results = extract_backbones(G_nx)

    # MDL backbone
    try:
        mdl_result = extract_mdl_backbone(G_nx)
        bb_results["mdl_global"] = {"mdl_edges": mdl_result["global"], "time": mdl_result["time"]}
        bb_results["mdl_local"] = {"mdl_edges": mdl_result["local"], "time": mdl_result["time"]}
    except Exception as e:
        logger.warning("  MDL FAILED: %s", e)

    # 5) Test each backbone
    # For statistical methods: test alpha = 0.01, 0.05, 0.1
    # For structural methods: test fraction = 0.1, 0.3, 0.5
    stat_methods = {"disparity", "noise_corrected", "ecm", "lans", "marginal_likelihood"}
    struct_methods = {"doubly_stochastic", "global_threshold"}

    all_results = {"baseline": baseline_results}

    for name, bb_data in bb_results.items():
        if "error" in bb_data:
            logger.info("\nSkipping %s (failed)", name)
            continue

        # MDL backbones: already filtered
        if name.startswith("mdl_"):
            logger.info("\n" + "-" * 70)
            logger.info("BACKBONE: %s (parameter-free)", name)

            mdl_edges = bb_data["mdl_edges"]
            G_bb = mdl_edges_to_networkx(mdl_edges, n_nodes)
            n_bb_edges = G_bb.number_of_edges()
            n_bb_nodes = G_bb.number_of_nodes()
            reduction = 1 - n_bb_edges / n_edges if n_edges > 0 else 0

            logger.info("  Edges: %d → %d (%.1f%% removed), time=%.1fs",
                        n_edges, n_bb_edges, reduction * 100, bb_data["time"])

            if n_bb_edges == 0:
                logger.info("  No edges in backbone, skipping")
                continue

            g_bb = networkx_to_igraph(G_bb)
            gcc_ids = g_bb.connected_components().giant().vs.indices
            g_bb_gcc = g_bb.subgraph(gcc_ids)
            logger.info("  GCC: %d nodes, %d edges", g_bb_gcc.vcount(), g_bb_gcc.ecount())

            results = {}
            for gamma in gammas:
                try:
                    part = run_leiden(g_bb_gcc, gamma=gamma)
                    metrics = compute_metrics(g_bb_gcc, part)
                    results[gamma] = metrics
                    logger.info("    γ=%.3f: %d clusters, cov=%.1f%%, max=%d, dens=%.4f",
                                gamma, metrics["n_clusters"], metrics["coverage"] * 100,
                                metrics["max_size"], metrics["avg_internal_density"])
                except Exception as e:
                    logger.warning("    γ=%.3f: FAILED: %s", gamma, e)

            all_results[name] = results
            continue

        # netbone methods
        if name in stat_methods:
            alphas = [0.01, 0.05, 0.1]
        elif name in struct_methods:
            alphas = [0.1, 0.3, 0.5]  # fraction
        else:
            continue

        for alpha in alphas:
            label = f"{name}_a{alpha}"
            logger.info("\n" + "-" * 70)
            logger.info("BACKBONE: %s (α/frac=%.2f)", name, alpha)

            try:
                G_filtered = filter_backbone(bb_data, alpha=alpha)
                if G_filtered is None or G_filtered.number_of_edges() == 0:
                    logger.info("  No edges after filtering, skipping")
                    continue

                n_bb_edges = G_filtered.number_of_edges()
                reduction = 1 - n_bb_edges / n_edges if n_edges > 0 else 0
                logger.info("  Edges: %d → %d (%.1f%% removed)",
                            n_edges, n_bb_edges, reduction * 100)

                g_bb = networkx_to_igraph(G_filtered)
                gcc_ids = g_bb.connected_components().giant().vs.indices
                g_bb_gcc = g_bb.subgraph(gcc_ids)
                logger.info("  GCC: %d nodes, %d edges", g_bb_gcc.vcount(), g_bb_gcc.ecount())

                results = {}
                for gamma in gammas:
                    try:
                        part = run_leiden(g_bb_gcc, gamma=gamma)
                        metrics = compute_metrics(g_bb_gcc, part)
                        results[gamma] = metrics
                        logger.info("    γ=%.3f: %d clusters, cov=%.1f%%, max=%d, dens=%.4f",
                                    gamma, metrics["n_clusters"], metrics["coverage"] * 100,
                                    metrics["max_size"], metrics["avg_internal_density"])
                    except Exception as e:
                        logger.warning("    γ=%.3f: FAILED: %s", gamma, e)

                all_results[label] = results

            except Exception as e:
                logger.warning("  %s FAILED: %s", label, e)

    # 6) Summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 70)

    header = f"{'Method':<30} {'γ':>5} {'#Clust':>7} {'Cov%':>6} {'Max':>6} {'Med':>6} {'Dens':>7} {'Mod':>7}"
    logger.info(header)
    logger.info("-" * len(header))

    for method_name, method_results in all_results.items():
        for gamma in gammas:
            if gamma in method_results:
                m = method_results[gamma]
                logger.info(
                    f"{method_name:<30} {gamma:>5.3f} {m['n_clusters']:>7d} "
                    f"{m['coverage']*100:>5.1f}% {m['max_size']:>6d} "
                    f"{m['median_size']:>6.0f} {m['avg_internal_density']:>7.4f} "
                    f"{m['modularity']:>7.4f}"
                )

    # 7) Timing summary
    logger.info("\n" + "=" * 70)
    logger.info("BACKBONE EXTRACTION TIMES")
    logger.info("=" * 70)
    for name, data in bb_results.items():
        t = data.get("time", 0)
        err = data.get("error", "")
        status = f"FAILED: {err}" if err else f"{t:.1f}s"
        logger.info("  %-25s %s", name, status)


if __name__ == "__main__":
    main()
