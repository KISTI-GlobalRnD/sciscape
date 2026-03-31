#!/usr/bin/env python3
"""Evaluate BOTH stability AND quality (subfield alignment) across link types.

This addresses the key question: Boyack says BC > DC for quality,
but DC > BC for stability. Are both true simultaneously?

Quality metrics (using subfield labels as ground truth):
- NMI: Normalized Mutual Information between clustering and subfields
- Purity: Average fraction of majority subfield per cluster
- V-measure: Harmonic mean of homogeneity and completeness

Usage:
    python scripts/eval_quality_vs_stability.py --field 12
    python scripts/eval_quality_vs_stability.py --field 12 --backbone topk --backbone-k 30
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
from sklearn.metrics import (
    normalized_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qvs")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges"
DATA_ROOT = Path.home() / "Desktop/HDD/local_map_analysis_data/processed/outputs/data"
NODE_DIR = DATA_ROOT / "oa26_gcc_only_k30"


def load_subfield_labels(field_id: int) -> dict[str, str]:
    """Load work_id → subfield_id mapping."""
    nodes = pl.read_parquet(
        NODE_DIR / f"field_{field_id}_nodes_oa_meta_gcc_k30.parquet",
        columns=["work_id", "subfield_id"],
    )
    # Each work_id appears once per subfield; take the first (most nodes have 1)
    # For multi-subfield nodes, pick the first one (arbitrary but consistent)
    mapping = {}
    for wid, sfid in zip(nodes["work_id"].to_list(), nodes["subfield_id"].to_list()):
        if wid not in mapping:
            mapping[wid] = sfid
    return mapping


def load_graph(field_id: int, link_type: str):
    """Load edge list → igraph, GCC. Returns (graph, work_ids_in_graph)."""
    path = EDGE_DIR / f"field_{field_id}" / f"{link_type}.parquet"
    df = pl.read_parquet(path)

    all_ids = pl.concat([df["src"], df["dst"]]).unique().sort().to_list()
    id2idx = {wid: i for i, wid in enumerate(all_ids)}

    src = [id2idx[s] for s in df["src"].to_list()]
    dst = [id2idx[d] for d in df["dst"].to_list()]
    weights = df["weight"].to_list()

    g = ig.Graph(n=len(all_ids), edges=list(zip(src, dst)), directed=False)
    g.es["weight"] = weights
    g.vs["work_id"] = all_ids
    g = g.simplify(combine_edges="max")

    # GCC
    gcc_idx = g.connected_components().giant().vs.indices
    g = g.subgraph(gcc_idx)
    work_ids = [g.vs[i]["work_id"] for i in range(g.vcount())]
    return g, work_ids


def auto_gamma(g: ig.Graph) -> float:
    weights = np.array(g.es["weight"])
    n, m = g.vcount(), g.ecount()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    gamma = float(np.median(weights)) * density * 2.0
    return max(min(gamma, 0.5), 1e-5)


def apply_backbone(g: ig.Graph, method: str, **kwargs) -> ig.Graph:
    """Apply backbone extraction."""
    if method == "topk":
        k = kwargs.get("k", 30)
        n = g.vcount()
        kept = set()
        for v in range(n):
            nbrs = g.neighbors(v)
            if not nbrs:
                continue
            ws = [(nb, g.es[g.get_eid(v, nb)]["weight"]) for nb in nbrs]
            ws.sort(key=lambda x: -x[1])
            for nb, w in ws[:k]:
                kept.add((min(v, nb), max(v, nb)))
        keep_eids = []
        for eid, e in enumerate(g.es):
            if (min(e.source, e.target), max(e.source, e.target)) in kept:
                keep_eids.append(eid)
        g = g.subgraph_edges(keep_eids, delete_vertices=False)
    return g


def evaluate_one(
    field_id: int, link_type: str, sf_labels: dict[str, str],
    n_runs: int = 30, backbone: str | None = None, **bb_kwargs,
):
    """Evaluate one link type: stability + quality."""
    t0 = time.time()
    g, work_ids = load_graph(field_id, link_type)

    if backbone:
        g = apply_backbone(g, backbone, **bb_kwargs)
        gcc_idx = g.connected_components().giant().vs.indices
        g = g.subgraph(gcc_idx)
        work_ids = [g.vs[i]["work_id"] for i in range(g.vcount())]

    n = g.vcount()
    if n < 10:
        return None

    gamma = auto_gamma(g)
    log.info("  %s: %d nodes, %d edges, γ=%.6f", link_type, n, g.ecount(), gamma)

    # Ground truth subfield labels for nodes in graph
    sf_true = [sf_labels.get(wid, "unknown") for wid in work_ids]

    # Run ensemble
    memberships = []
    qualities_list = []
    sf_nmis = []
    sf_vmeasures = []
    sf_purities = []

    for seed in range(n_runs):
        part = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma, weights="weight", seed=seed,
        )
        mem = part.membership
        memberships.append(mem)
        qualities_list.append(part.quality())

        # Quality: cluster vs subfield
        sf_nmis.append(normalized_mutual_info_score(sf_true, mem))
        sf_vmeasures.append(v_measure_score(sf_true, mem))

        # Purity
        cnts = Counter(zip(mem, sf_true))
        cluster_sizes = Counter(mem)
        purity_sum = 0
        for cid, size in cluster_sizes.items():
            max_sf = max(cnts.get((cid, sf), 0) for sf in set(sf_true))
            purity_sum += max_sf
        sf_purities.append(purity_sum / n)

    # Stability: pairwise NMI between runs
    mem_arr = np.array(memberships)
    sample = min(20, n_runs)
    idx_s = np.random.RandomState(42).choice(n_runs, sample, replace=False)
    run_nmis = []
    for i in range(len(idx_s)):
        for j in range(i + 1, len(idx_s)):
            run_nmis.append(normalized_mutual_info_score(
                memberships[idx_s[i]], memberships[idx_s[j]]))

    # Edge co-membership
    edge_list = g.get_edgelist()
    edges_arr = np.array(edge_list)
    src, dst = edges_arr[:, 0], edges_arr[:, 1]
    co_rates = np.mean(mem_arr[:, src] == mem_arr[:, dst], axis=0)

    # Node stability
    node_stability = np.zeros(n)
    node_degree = np.zeros(n)
    np.add.at(node_stability, src, co_rates)
    np.add.at(node_stability, dst, co_rates)
    np.add.at(node_degree, src, 1.0)
    np.add.at(node_degree, dst, 1.0)
    mask = node_degree > 0
    node_stability[mask] /= node_degree[mask]
    node_stability[~mask] = 1.0

    n_clusters = [len(set(m)) for m in memberships]

    return {
        "n_nodes": n,
        "n_edges": g.ecount(),
        "gamma": gamma,
        "n_clusters": float(np.mean(n_clusters)),
        # Quality metrics (cluster vs subfield)
        "sf_nmi": float(np.mean(sf_nmis)),
        "sf_nmi_std": float(np.std(sf_nmis)),
        "sf_vmeasure": float(np.mean(sf_vmeasures)),
        "sf_purity": float(np.mean(sf_purities)),
        # Stability metrics
        "run_nmi": float(np.mean(run_nmis)),
        "node_stab": float(node_stability.mean()),
        "hub_stab": float(node_stability[np.argsort(np.array(g.degree()))[-max(1, n//10):]].mean()),
        "unstable_pct": float(np.sum(node_stability < 0.5) / n),
        "edge_co": float(co_rates.mean()),
        "time": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=int, default=12)
    parser.add_argument("--link-types", nargs="*", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--backbone-k", type=int, default=30)
    parser.add_argument("--n-runs", type=int, default=30)
    args = parser.parse_args()

    sf_labels = load_subfield_labels(args.field)
    n_sf = len(set(sf_labels.values()))
    log.info("Field %d: %d nodes, %d subfields as ground truth", args.field, len(sf_labels), n_sf)

    edge_dir = EDGE_DIR / f"field_{args.field}"
    if args.link_types:
        link_types = args.link_types
    else:
        link_types = sorted(p.stem for p in edge_dir.glob("*.parquet") if p.stem != "node_mapping")

    results = {}
    for lt in link_types:
        label = lt
        bb_kwargs = {}
        if args.backbone == "topk":
            bb_kwargs = {"k": args.backbone_k}
            label = f"{lt}+topk{args.backbone_k}"
        r = evaluate_one(args.field, lt, sf_labels, args.n_runs, args.backbone, **bb_kwargs)
        results[label] = r

    # Print results
    print(f"\n{'='*120}")
    print(f"QUALITY vs STABILITY — Field {args.field} ({n_sf} subfields as ground truth)")
    print(f"{'='*120}")
    print(f"{'Link Type':<28} {'Nodes':>6} {'Edges':>8} {'#Clust':>6} │"
          f" {'SF_NMI':>7} {'SF_Vmsr':>7} {'Purity':>7} │"
          f" {'RunNMI':>7} {'NodeSt':>7} {'HubSt':>7} {'Unstbl':>7} │ {'Time':>5}")
    print("-" * 120)

    for name, r in sorted(results.items()):
        if r is None:
            print(f"{name:<28} SKIPPED")
            continue
        print(f"{name:<28} {r['n_nodes']:>6} {r['n_edges']:>8} {r['n_clusters']:>6.0f} │"
              f" {r['sf_nmi']:>7.4f} {r['sf_vmeasure']:>7.4f} {r['sf_purity']:>7.4f} │"
              f" {r['run_nmi']:>7.4f} {r['node_stab']:>7.4f} {r['hub_stab']:>7.4f}"
              f" {r['unstable_pct']*100:>6.1f}% │ {r['time']:>4.0f}s")


if __name__ == "__main__":
    main()
