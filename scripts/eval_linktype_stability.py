#!/usr/bin/env python3
"""Evaluate Leiden ensemble stability across link types, normalizations, and backbones.

For each edge set:
  1. Build igraph from edge list
  2. Run Leiden CPM ensemble (N runs)
  3. Measure: NMI pairwise, node stability tiers, hub instability
  4. Optionally apply backbone (top-k, disparity) before clustering

Usage:
    python scripts/eval_linktype_stability.py --field 34
    python scripts/eval_linktype_stability.py --field 34 --link-types dc_binary bc_cosine
    python scripts/eval_linktype_stability.py --field 34 --backbone topk --backbone-k 30
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import polars as pl
from sklearn.metrics import normalized_mutual_info_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges"

N_RUNS = 50
GAMMA_AUTO = True  # auto-select gamma per network


def load_graph(field_id: int, link_type: str) -> ig.Graph:
    """Load edge list → igraph, GCC only."""
    path = EDGE_DIR / f"field_{field_id}" / f"{link_type}.parquet"
    df = pl.read_parquet(path)

    # Build node mapping
    all_ids = pl.concat([df["src"], df["dst"]]).unique().sort().to_list()
    id2idx = {wid: i for i, wid in enumerate(all_ids)}

    src = [id2idx[s] for s in df["src"].to_list()]
    dst = [id2idx[d] for d in df["dst"].to_list()]
    weights = df["weight"].to_list()

    g = ig.Graph(n=len(all_ids), edges=list(zip(src, dst)), directed=False)
    g.es["weight"] = weights
    g = g.simplify(combine_edges="max")

    # GCC
    gcc = g.connected_components().giant()
    gcc_g = g.subgraph(gcc.vs.indices)
    log.info("  %s: %d nodes, %d edges (GCC: %d/%d)",
             link_type, g.vcount(), g.ecount(), gcc_g.vcount(), gcc_g.ecount())
    return gcc_g


def auto_gamma(g: ig.Graph) -> float:
    """Estimate CPM resolution from network properties."""
    weights = np.array(g.es["weight"])
    n = g.vcount()
    m = g.ecount()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0

    # Heuristic: median_weight * density * scale_factor
    # Tuned to give reasonable cluster counts (not all singletons, not 1 cluster)
    gamma = float(np.median(weights)) * density * 2.0
    gamma = max(gamma, 1e-5)
    gamma = min(gamma, 0.5)

    log.info("  auto γ = %.6f (median_w=%.4f, density=%.6f)",
             gamma, np.median(weights), density)
    return gamma


def apply_backbone(g: ig.Graph, method: str, **kwargs) -> ig.Graph:
    """Apply backbone extraction, return filtered graph."""
    n = g.vcount()
    weights = np.array(g.es["weight"])

    if method == "topk":
        k = kwargs.get("k", 30)
        kept = set()
        for v in range(n):
            nbrs = g.neighbors(v)
            if len(nbrs) == 0:
                continue
            ws = [(nb, g.es[g.get_eid(v, nb)]["weight"]) for nb in nbrs]
            ws.sort(key=lambda x: -x[1])
            for nb, w in ws[:k]:
                kept.add((min(v, nb), max(v, nb)))
        # Map to edge ids
        keep_eids = []
        for eid, e in enumerate(g.es):
            key = (min(e.source, e.target), max(e.source, e.target))
            if key in kept:
                keep_eids.append(eid)
        g_f = g.subgraph_edges(keep_eids, delete_vertices=False)
        log.info("  backbone topk(k=%d): %d → %d edges", k, g.ecount(), g_f.ecount())
        return g_f

    elif method == "threshold":
        pct = kwargs.get("percentile", 30)
        thr = np.percentile(weights, pct)
        keep_eids = [i for i, w in enumerate(weights) if w > thr]
        g_f = g.subgraph_edges(keep_eids, delete_vertices=False)
        log.info("  backbone threshold(p%d=%.6f): %d → %d edges",
                 pct, thr, g.ecount(), g_f.ecount())
        return g_f

    return g


def leiden_ensemble(g: ig.Graph, gamma: float, n_runs: int = N_RUNS):
    """Run Leiden CPM ensemble, return memberships and quality stats."""
    memberships = []
    qualities = []
    n_clusters_list = []

    for seed in range(n_runs):
        part = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma, weights="weight", seed=seed,
        )
        mem = part.membership
        memberships.append(mem)
        qualities.append(part.quality())
        big = sum(1 for cnt in Counter(mem).values() if cnt >= 5)
        n_clusters_list.append(big)

    return memberships, qualities, n_clusters_list


def compute_stability(memberships: list, g: ig.Graph):
    """Compute stability metrics from ensemble (vectorized)."""
    n = g.vcount()
    n_runs = len(memberships)
    mem_arr = np.array(memberships)  # shape (n_runs, n)

    # ── Edge co-membership (vectorized) ──
    edge_list = g.get_edgelist()
    edges_arr = np.array(edge_list)  # shape (n_edges, 2)
    src = edges_arr[:, 0]
    dst = edges_arr[:, 1]

    # For each run, check if src and dst have same membership
    # mem_arr[:, src] shape: (n_runs, n_edges), mem_arr[:, dst] shape: (n_runs, n_edges)
    co_rates = np.mean(mem_arr[:, src] == mem_arr[:, dst], axis=0)

    # ── Node stability (fully vectorized via np.add.at) ──
    node_stability = np.zeros(n)
    node_degree = np.zeros(n)
    np.add.at(node_stability, src, co_rates)
    np.add.at(node_stability, dst, co_rates)
    np.add.at(node_degree, src, 1.0)
    np.add.at(node_degree, dst, 1.0)
    mask = node_degree > 0
    node_stability[mask] /= node_degree[mask]
    node_stability[~mask] = 1.0

    # Stability tiers
    tiers = {
        "stable (>0.9)": np.sum(node_stability > 0.9) / n,
        "moderate (0.5-0.9)": np.sum((node_stability >= 0.5) & (node_stability <= 0.9)) / n,
        "unstable (<0.5)": np.sum(node_stability < 0.5) / n,
    }

    # ── Hub stability: top-degree nodes ──
    degrees = np.array(g.degree())
    top10_pct = max(1, int(n * 0.1))
    hub_idx = np.argsort(degrees)[-top10_pct:]
    hub_stability = node_stability[hub_idx].mean()

    edge_tiers = {
        "always (>0.9)": np.sum(co_rates > 0.9) / len(edge_list),
        "ambiguous (0.1-0.9)": np.sum((co_rates >= 0.1) & (co_rates <= 0.9)) / len(edge_list),
        "never (<0.1)": np.sum(co_rates < 0.1) / len(edge_list),
    }

    # ── Pairwise NMI ──
    sample = min(20, n_runs)
    idx_sample = np.random.RandomState(42).choice(n_runs, sample, replace=False)
    nmis = []
    for i in range(len(idx_sample)):
        for j in range(i + 1, len(idx_sample)):
            nmis.append(normalized_mutual_info_score(
                memberships[idx_sample[i]], memberships[idx_sample[j]]))
    nmis = np.array(nmis)

    return {
        "node_stability_mean": float(node_stability.mean()),
        "node_tiers": tiers,
        "hub_stability": float(hub_stability),
        "edge_tiers": edge_tiers,
        "edge_co_mean": float(co_rates.mean()),
        "nmi_mean": float(nmis.mean()),
        "nmi_std": float(nmis.std()),
        "nmi_min": float(nmis.min()),
    }


def evaluate(field_id: int, link_type: str, backbone: str | None = None, n_runs: int = 50, **bb_kwargs):
    """Full evaluation pipeline for one link type."""
    log.info("Evaluating field=%d, link=%s, backbone=%s", field_id, link_type, backbone)
    t0 = time.time()

    g = load_graph(field_id, link_type)
    if backbone:
        g = apply_backbone(g, backbone, **bb_kwargs)
        # Re-extract GCC after backbone
        gcc_idx = g.connected_components().giant().vs.indices
        g = g.subgraph(gcc_idx)
        log.info("  After backbone GCC: %d nodes, %d edges", g.vcount(), g.ecount())

    if g.ecount() == 0 or g.vcount() < 10:
        log.warning("  Graph too small, skipping")
        return None

    gamma = auto_gamma(g)
    memberships, qualities, n_clusters = leiden_ensemble(g, gamma, n_runs)

    stability = compute_stability(memberships, g)
    stability["gamma"] = gamma
    stability["n_nodes"] = g.vcount()
    stability["n_edges"] = g.ecount()
    stability["n_clusters_mean"] = float(np.mean(n_clusters))
    stability["quality_mean"] = float(np.mean(qualities))
    stability["quality_std"] = float(np.std(qualities))
    stability["time_sec"] = time.time() - t0

    return stability


def print_results(results: dict):
    """Print comparison table."""
    print(f"\n{'='*110}")
    print(f"{'Link Type':<25} {'Nodes':>6} {'Edges':>8} {'γ':>8} "
          f"{'#Clust':>6} {'NMI':>6} {'NodeStab':>8} {'HubStab':>8} "
          f"{'EdgeCo':>7} {'Stable%':>8} {'Unstbl%':>8} {'Time':>5}")
    print("-" * 110)

    for name, s in sorted(results.items()):
        if s is None:
            print(f"{name:<25} {'SKIPPED':>6}")
            continue
        print(f"{name:<25} {s['n_nodes']:>6} {s['n_edges']:>8} {s['gamma']:>8.5f} "
              f"{s['n_clusters_mean']:>6.0f} {s['nmi_mean']:>6.3f} "
              f"{s['node_stability_mean']:>8.3f} {s['hub_stability']:>8.3f} "
              f"{s['edge_co_mean']:>7.3f} "
              f"{s['node_tiers']['stable (>0.9)']*100:>7.1f}% "
              f"{s['node_tiers']['unstable (<0.5)']*100:>7.1f}% "
              f"{s['time_sec']:>5.0f}s")

    # Edge tier detail
    print(f"\n{'Link Type':<25} {'Edge always%':>12} {'Edge ambig%':>12} {'Edge never%':>12}")
    print("-" * 65)
    for name, s in sorted(results.items()):
        if s is None:
            continue
        print(f"{name:<25} "
              f"{s['edge_tiers']['always (>0.9)']*100:>11.1f}% "
              f"{s['edge_tiers']['ambiguous (0.1-0.9)']*100:>11.1f}% "
              f"{s['edge_tiers']['never (<0.1)']*100:>11.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Leiden stability per link type")
    parser.add_argument("--field", type=int, default=34)
    parser.add_argument("--link-types", nargs="*", default=None,
                        help="Which link types to evaluate. Default: all in directory")
    parser.add_argument("--backbone", choices=["topk", "threshold", None], default=None)
    parser.add_argument("--backbone-k", type=int, default=30)
    parser.add_argument("--backbone-pct", type=int, default=30)
    parser.add_argument("--n-runs", type=int, default=50)
    args = parser.parse_args()

    N_RUNS = args.n_runs

    # Discover available link types
    edge_dir = EDGE_DIR / f"field_{args.field}"
    if not edge_dir.exists():
        log.error("No data for field %d. Run build_linktype_edges.py first.", args.field)
        return

    if args.link_types:
        link_types = args.link_types
    else:
        link_types = sorted([
            p.stem for p in edge_dir.glob("*.parquet")
            if p.stem != "node_mapping"
        ])

    log.info("Field %d: evaluating %d link types: %s", args.field, len(link_types), link_types)

    results = {}
    for lt in link_types:
        bb_kwargs = {}
        label = lt
        if args.backbone == "topk":
            bb_kwargs = {"k": args.backbone_k}
            label = f"{lt}+topk{args.backbone_k}"
        elif args.backbone == "threshold":
            bb_kwargs = {"percentile": args.backbone_pct}
            label = f"{lt}+thr_p{args.backbone_pct}"

        results[label] = evaluate(
            args.field, lt,
            backbone=args.backbone, n_runs=N_RUNS, **bb_kwargs,
        )

    print_results(results)


if __name__ == "__main__":
    main()
