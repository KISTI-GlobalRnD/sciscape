#!/usr/bin/env python3
"""Evaluate individual & combined link-type clustering on oa26_gcc_only data.

For each edge set (individual DC/BC/CC + combinations):
  1. Build igraph from edge list (GCC only)
  2. Run Leiden CPM ensemble (N_RUNS seeds)
  3. Measure: pairwise NMI, node stability tiers, hub instability
  4. Compare across link types and combinations

Usage:
    # Field 15 (small, ~115K nodes) — quick test
    .venv/bin/python scripts/eval_linktype_gcc.py --field 15 --n-runs 10

    # Field 12 (large, ~753K nodes) — full evaluation
    .venv/bin/python scripts/eval_linktype_gcc.py --field 12

    # Specific link types only
    .venv/bin/python scripts/eval_linktype_gcc.py --field 15 --link-types bc_assoc_strength cc_assoc_strength

    # Skip combinations
    .venv/bin/python scripts/eval_linktype_gcc.py --field 15 --no-combo

    # Custom gamma
    .venv/bin/python scripts/eval_linktype_gcc.py --field 15 --gamma 1e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_gcc")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results_gcc"

# ── Best normalization per link type ──────────────────────────────
INDIVIDUAL_TYPES = {
    "dc_frac": "dc_fractional",
    "bc_as": "bc_assoc_strength",
    "cc_as": "cc_assoc_strength",
    "emb": "emb_full_knn30",
}

# All 6 layers including bg/nov split embeddings
ALL_LAYER_TYPES = {
    "DC": "dc_fractional",
    "BC": "bc_assoc_strength",
    "CC": "cc_assoc_strength",
    "Emb_full": "emb_full_knn30",
    "Emb_bg": "emb_bg_knn30",
    "Emb_nov": "emb_nov_knn30",
}

# Combination methods to test
COMBO_METHODS = ["sum", "max", "noisy_or"]


# ── Backbone (top-k per node) ─────────────────────────────────────

def apply_topk_backbone(M: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Keep only top-k edges per node by weight (symmetric).

    For each row, keep the k largest weights. The result is the union
    of top-k from both endpoints (symmetric).
    """
    n = M.shape[0]
    M_csr = M.tocsr()
    rows, cols, data = [], [], []

    for i in range(n):
        start, end = M_csr.indptr[i], M_csr.indptr[i + 1]
        if end - start <= k:
            # Keep all
            cols_i = M_csr.indices[start:end]
            data_i = M_csr.data[start:end]
        else:
            # Top-k by weight
            local_data = M_csr.data[start:end]
            local_cols = M_csr.indices[start:end]
            topk_idx = np.argpartition(local_data, -k)[-k:]
            cols_i = local_cols[topk_idx]
            data_i = local_data[topk_idx]

        for c, w in zip(cols_i, data_i):
            rows.append(i)
            cols.append(c)
            data.append(w)

    M_filtered = sparse.csr_matrix(
        (np.array(data), (np.array(rows), np.array(cols))),
        shape=(n, n),
    )
    # Symmetrize: union of top-k from both directions
    M_sym = M_filtered.maximum(M_filtered.T)
    return M_sym


def _df_to_sparse_topk(
    df: pl.DataFrame,
    id2idx: dict[str, int],
    n: int,
    topk: int | None = None,
) -> sparse.csr_matrix:
    """Convert edge DataFrame to sparse matrix, optionally apply top-k."""
    src = np.array([id2idx[x] for x in df["uid1"].to_list()], dtype=np.int32)
    dst = np.array([id2idx[x] for x in df["uid2"].to_list()], dtype=np.int32)
    w = df["rel_sum2"].to_numpy().astype(np.float32)

    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    data = np.concatenate([w, w])
    M = sparse.csr_matrix((data, (row, col)), shape=(n, n))

    if topk is not None:
        before = M.nnz // 2
        M = apply_topk_backbone(M, topk)
        after = M.nnz // 2
        log.info("    top-k(%d): %d → %d edges (%.1f%%)", topk, before, after, after / before * 100)

    return M


# ── Graph construction ────────────────────────────────────────────

def load_graph(
    field_id: int,
    link_type: str,
    topk: int | None = None,
    node_subset: set[str] | None = None,
) -> tuple[ig.Graph, list[str]]:
    """Load edge list → igraph (GCC). Returns (graph, node_ids_in_gcc).

    If node_subset is given, filter edges to only those connecting nodes
    in the subset BEFORE building the graph.
    """
    path = EDGE_DIR / f"field_{field_id}" / f"{link_type}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pl.read_parquet(path)

    # Filter to node subset if given
    if node_subset is not None:
        subset_series = pl.Series("_s", sorted(node_subset))
        df = df.filter(
            pl.col("uid1").is_in(subset_series) & pl.col("uid2").is_in(subset_series)
        )
        log.info("    node_subset filter: %d edges retained", df.height)

    # Columns: uid1, uid2, rel_sum2
    all_ids = (
        pl.concat([df["uid1"].alias("id"), df["uid2"].alias("id")])
        .unique()
        .sort()
        .to_list()
    )
    id2idx = {wid: i for i, wid in enumerate(all_ids)}
    n = len(all_ids)

    M = _df_to_sparse_topk(df, id2idx, n, topk=topk)

    # Convert to igraph
    upper = sparse.triu(M, k=1).tocoo()
    mask = upper.data > 0
    edges = list(zip(upper.row[mask].tolist(), upper.col[mask].tolist()))
    weights = upper.data[mask].astype(np.float64).tolist()

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    g = g.simplify(combine_edges="max")

    # Extract GCC
    comps = g.connected_components()
    gcc_idx = comps.giant().vs.indices if hasattr(comps.giant(), 'vs') else list(range(g.vcount()))
    g_gcc = g.subgraph(gcc_idx)

    gcc_node_ids = [all_ids[i] for i in gcc_idx]
    log.info("  %s: %d nodes, %d edges → GCC: %d nodes, %d edges",
             link_type, n, len(edges), g_gcc.vcount(), g_gcc.ecount())
    return g_gcc, gcc_node_ids


def load_and_combine(
    field_id: int,
    link_types: list[str],
    method: str,
    topk: int | None = None,
    node_subset: set[str] | None = None,
) -> tuple[ig.Graph, list[str]]:
    """Load multiple edge sets, combine, return igraph (GCC)."""
    # Union of all node IDs across types
    all_dfs = {}
    all_node_ids = set()
    for lt in link_types:
        path = EDGE_DIR / f"field_{field_id}" / f"{lt}.parquet"
        df = pl.read_parquet(path)
        # Filter to node subset if given
        if node_subset is not None:
            subset_series = pl.Series("_s", sorted(node_subset))
            df = df.filter(
                pl.col("uid1").is_in(subset_series) & pl.col("uid2").is_in(subset_series)
            )
        all_dfs[lt] = df
        all_node_ids.update(df["uid1"].to_list())
        all_node_ids.update(df["uid2"].to_list())

    all_node_ids = sorted(all_node_ids)
    id2idx = {wid: i for i, wid in enumerate(all_node_ids)}
    n = len(all_node_ids)

    # Build sparse matrices (top-k filtered per layer, then normalized to [0, 1])
    matrices = []
    for lt in link_types:
        effective_k = _effective_topk(lt, topk)
        log.info("    Loading %s for combination (topk=%s)...", lt, effective_k)
        M = _df_to_sparse_topk(all_dfs[lt], id2idx, n, topk=effective_k)

        # Normalize to [0, 1]
        mx = M.data.max()
        if mx > 0:
            M.data = M.data / mx
        matrices.append(M)

    # Combine
    if method == "sum":
        C = matrices[0]
        for M in matrices[1:]:
            C = C + M
    elif method == "max":
        C = matrices[0]
        for M in matrices[1:]:
            C = C.maximum(M)
    elif method == "noisy_or":
        # 1 - prod(1 - w_i) via log trick
        # For sparse: iterate COO
        C = matrices[0].copy()
        for M in matrices[1:]:
            # noisy_or: result = 1 - (1 - result) * (1 - M)
            # = result + M - result * M
            product = C.multiply(M)
            C = C + M - product
    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert to igraph
    upper = sparse.triu(C, k=1).tocoo()
    mask = upper.data > 0
    edges = list(zip(upper.row[mask].tolist(), upper.col[mask].tolist()))
    weights = upper.data[mask].tolist()

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    g = g.simplify(combine_edges="max")

    # GCC
    comps = g.connected_components()
    gcc_vs = comps.giant().vs.indices
    g_gcc = g.subgraph(gcc_vs)
    gcc_ids = [all_node_ids[i] for i in gcc_vs]

    combo_name = "+".join(lt.split("_")[0] for lt in link_types)
    log.info("  combo %s (%s): %d nodes, %d edges → GCC: %d, %d",
             combo_name, method, n, len(edges), g_gcc.vcount(), g_gcc.ecount())
    return g_gcc, gcc_ids


# ── Gamma estimation ─────────────────────────────────────────────

def auto_gamma(g: ig.Graph) -> float:
    """Estimate CPM resolution from network properties."""
    weights = np.array(g.es["weight"])
    n = g.vcount()
    m = g.ecount()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0

    gamma = float(np.median(weights)) * density * 2.0
    gamma = max(gamma, 1e-6)
    gamma = min(gamma, 0.1)
    log.info("  auto γ = %.6e (median_w=%.4f, density=%.2e, n=%d, m=%d)",
             gamma, np.median(weights), density, n, m)
    return gamma


# ── Leiden ensemble ──────────────────────────────────────────────

def leiden_ensemble(
    g: ig.Graph,
    gamma: float,
    n_runs: int = 50,
) -> tuple[list[list[int]], list[float], list[int]]:
    """Run Leiden CPM ensemble, return (memberships, qualities, n_clusters)."""
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


# ── Stability metrics ────────────────────────────────────────────

def compute_stability(memberships: list[list[int]], g: ig.Graph) -> dict:
    """Compute stability metrics from ensemble."""
    n = g.vcount()
    n_runs = len(memberships)
    mem_arr = np.array(memberships)

    # Edge co-membership
    edges_arr = np.array(g.get_edgelist())
    src = edges_arr[:, 0]
    dst = edges_arr[:, 1]
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

    tiers = {
        "stable_pct": float(np.sum(node_stability > 0.9) / n),
        "moderate_pct": float(np.sum((node_stability >= 0.5) & (node_stability <= 0.9)) / n),
        "unstable_pct": float(np.sum(node_stability < 0.5) / n),
    }

    # Hub stability (top 10%)
    degrees = np.array(g.degree())
    top10_pct = max(1, int(n * 0.1))
    hub_idx = np.argsort(degrees)[-top10_pct:]
    hub_stability = float(node_stability[hub_idx].mean())

    # Edge tiers
    edge_tiers = {
        "always_pct": float(np.sum(co_rates > 0.9) / len(co_rates)),
        "ambiguous_pct": float(np.sum((co_rates >= 0.1) & (co_rates <= 0.9)) / len(co_rates)),
        "never_pct": float(np.sum(co_rates < 0.1) / len(co_rates)),
    }

    # Pairwise NMI (sample up to 20 pairs)
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
        "hub_stability": hub_stability,
        "edge_tiers": edge_tiers,
        "edge_co_mean": float(co_rates.mean()),
        "nmi_mean": float(nmis.mean()),
        "nmi_std": float(nmis.std()),
        "nmi_min": float(nmis.min()),
    }


# ── Single evaluation ────────────────────────────────────────────

def _effective_topk(link_type: str, topk: int | None) -> int | None:
    """DC edges have uniform weights → top-k is meaningless. Skip for DC."""
    if topk is None:
        return None
    if link_type.startswith("dc_"):
        log.info("  Skipping top-k for DC (uniform weights)")
        return None
    return topk


def evaluate_single(
    field_id: int,
    link_type: str,
    gamma: float | None = None,
    n_runs: int = 50,
    topk: int | None = None,
    node_subset: set[str] | None = None,
) -> dict | None:
    """Full evaluation for one individual link type."""
    effective_k = _effective_topk(link_type, topk)
    log.info("── Evaluating field=%d, link=%s (topk=%s, subset=%s) ──",
             field_id, link_type, effective_k,
             f"{len(node_subset)}nodes" if node_subset else "all")
    t0 = time.time()

    try:
        g, node_ids = load_graph(field_id, link_type, topk=effective_k,
                                 node_subset=node_subset)
    except FileNotFoundError as e:
        log.warning("  %s", e)
        return None

    if g.ecount() == 0 or g.vcount() < 10:
        log.warning("  Graph too small, skipping")
        return None

    gamma_used = gamma or auto_gamma(g)
    memberships, qualities, n_clusters = leiden_ensemble(g, gamma_used, n_runs)
    stability = compute_stability(memberships, g)

    result = {
        **stability,
        "gamma": gamma_used,
        "n_nodes": g.vcount(),
        "n_edges": g.ecount(),
        "n_clusters_mean": float(np.mean(n_clusters)),
        "n_clusters_std": float(np.std(n_clusters)),
        "quality_mean": float(np.mean(qualities)),
        "time_sec": time.time() - t0,
    }
    return result


def evaluate_combo(
    field_id: int,
    link_types: list[str],
    method: str,
    gamma: float | None = None,
    n_runs: int = 50,
    topk: int | None = None,
    node_subset: set[str] | None = None,
) -> dict | None:
    """Full evaluation for a combined link type."""
    combo_name = "+".join(lt.split("_")[0] for lt in link_types) + f"_{method}"
    log.info("── Evaluating field=%d, combo=%s (topk=%s) ──", field_id, combo_name, topk)
    t0 = time.time()

    g, node_ids = load_and_combine(field_id, link_types, method, topk=topk,
                                   node_subset=node_subset)

    if g.ecount() == 0 or g.vcount() < 10:
        log.warning("  Graph too small, skipping")
        return None

    gamma_used = gamma or auto_gamma(g)
    memberships, qualities, n_clusters = leiden_ensemble(g, gamma_used, n_runs)
    stability = compute_stability(memberships, g)

    result = {
        **stability,
        "gamma": gamma_used,
        "n_nodes": g.vcount(),
        "n_edges": g.ecount(),
        "n_clusters_mean": float(np.mean(n_clusters)),
        "n_clusters_std": float(np.std(n_clusters)),
        "quality_mean": float(np.mean(qualities)),
        "time_sec": time.time() - t0,
    }
    return result


# ── Cross-type AMI matrix ────────────────────────────────────────

def compute_cross_ami(
    field_id: int,
    link_types: dict[str, str],
    gamma: float | None = None,
    topk: int | None = None,
    node_subset: set[str] | None = None,
) -> pl.DataFrame:
    """Compute AMI matrix between best partitions of each link type.

    Uses seed=0 partition from each type.
    """
    from sklearn.metrics import adjusted_mutual_info_score

    log.info("── Computing cross-type AMI matrix ──")
    partitions = {}
    common_nodes = None  # Track node intersection

    for label, filename in link_types.items():
        try:
            effective_k = _effective_topk(filename, topk)
            g, node_ids = load_graph(field_id, filename, topk=effective_k,
                                     node_subset=node_subset)
        except FileNotFoundError:
            continue

        gamma_used = gamma or auto_gamma(g)
        part = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma_used, weights="weight", seed=0,
        )
        # Map: node_id → cluster_id
        id_to_cluster = {nid: part.membership[i] for i, nid in enumerate(node_ids)}
        partitions[label] = id_to_cluster

        if common_nodes is None:
            common_nodes = set(node_ids)
        else:
            common_nodes &= set(node_ids)

    if len(partitions) < 2:
        return pl.DataFrame()

    common = sorted(common_nodes)
    log.info("  Common nodes across all types: %d", len(common))

    names = list(partitions.keys())
    ami_matrix = np.zeros((len(names), len(names)))

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                ami_matrix[i][j] = 1.0
            elif i < j:
                labels_a = [partitions[a][n] for n in common]
                labels_b = [partitions[b][n] for n in common]
                ami = adjusted_mutual_info_score(labels_a, labels_b)
                ami_matrix[i][j] = ami
                ami_matrix[j][i] = ami

    # Build DataFrame
    rows = []
    for i, a in enumerate(names):
        row = {"type": a}
        for j, b in enumerate(names):
            row[b] = round(ami_matrix[i][j], 4)
        rows.append(row)

    return pl.DataFrame(rows)


# ── Output ────────────────────────────────────────────────────────

def print_results(results: dict[str, dict], field_id: int):
    """Print comparison table."""
    print(f"\n{'='*120}")
    print(f"  LINK-TYPE EVALUATION — Field {field_id} (oa26_gcc_only)")
    print(f"{'='*120}")
    print(f"{'Type':<28} {'Nodes':>8} {'Edges':>10} {'γ':>10} "
          f"{'#Clust':>8} {'NMI':>6} {'NodeStab':>8} {'HubStab':>8} "
          f"{'Stable%':>8} {'Unstbl%':>8} {'Time':>6}")
    print("-" * 120)

    for name, s in results.items():
        if s is None:
            print(f"{name:<28} {'SKIPPED':>8}")
            continue
        print(f"{name:<28} {s['n_nodes']:>8,} {s['n_edges']:>10,} "
              f"{s['gamma']:>10.2e} "
              f"{s['n_clusters_mean']:>7.0f}±{s['n_clusters_std']:<3.0f}"
              f"{s['nmi_mean']:>6.3f} "
              f"{s['node_stability_mean']:>8.3f} {s['hub_stability']:>8.3f} "
              f"{s['node_tiers']['stable_pct']*100:>7.1f}% "
              f"{s['node_tiers']['unstable_pct']*100:>7.1f}% "
              f"{s['time_sec']:>5.0f}s")

    # Edge tier detail
    print(f"\n{'Type':<28} {'Edge always%':>12} {'Edge ambig%':>12} {'Edge never%':>12}")
    print("-" * 68)
    for name, s in results.items():
        if s is None:
            continue
        print(f"{name:<28} "
              f"{s['edge_tiers']['always_pct']*100:>11.1f}% "
              f"{s['edge_tiers']['ambiguous_pct']*100:>11.1f}% "
              f"{s['edge_tiers']['never_pct']*100:>11.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate link-type clustering quality on oa26_gcc_only")
    parser.add_argument("--field", type=int, required=True,
                        help="Field ID (12 or 15)")
    parser.add_argument("--link-types", nargs="*", default=None,
                        help="Specific link types to evaluate (filenames without .parquet)")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Number of Leiden ensemble runs (default: 50)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Fixed gamma (default: auto-estimate)")
    parser.add_argument("--no-combo", action="store_true",
                        help="Skip combination evaluation")
    parser.add_argument("--no-ami", action="store_true",
                        help="Skip cross-type AMI matrix")
    parser.add_argument("--combo-methods", nargs="*", default=COMBO_METHODS,
                        help="Combination methods (default: sum max noisy_or)")
    parser.add_argument("--topk", type=int, default=None,
                        help="Top-k backbone per node (default: None = no filtering)")
    parser.add_argument("--common-subset", choices=["cc", "bc", "dc", None], default=None,
                        help="Restrict all layers to the GCC of this type's node set")
    parser.add_argument("--all-layers", action="store_true",
                        help="Use all 6 layers (DC/BC/CC/Emb_full/Emb_bg/Emb_nov) for AMI matrix")
    parser.add_argument("--ami-only", action="store_true",
                        help="Skip individual/combo evaluation, only compute cross-AMI matrix")
    args = parser.parse_args()

    field_id = args.field
    topk = args.topk
    edge_dir = EDGE_DIR / f"field_{field_id}"
    if not edge_dir.exists():
        log.error("No data for field %d at %s", field_id, edge_dir)
        return

    if topk:
        log.info("Using top-k backbone: k=%d (skipped for DC)", topk)

    # ── 0. Determine common node subset ──────────────────────────
    node_subset: set[str] | None = None
    if args.common_subset:
        # Find the matching filename
        subset_map = {"dc": "dc_fractional", "bc": "bc_assoc_strength", "cc": "cc_assoc_strength"}
        subset_file = subset_map[args.common_subset]
        log.info("Building common node subset from %s GCC...", subset_file)
        g_ref, ref_ids = load_graph(field_id, subset_file)
        node_subset = set(ref_ids)
        log.info("Common node subset: %d nodes (from %s GCC)", len(node_subset), args.common_subset)

    # ── 1. Individual link types ──────────────────────────────────
    if args.link_types:
        types_to_eval = {lt: lt for lt in args.link_types}
    else:
        types_to_eval = INDIVIDUAL_TYPES

    results = {}
    if not args.ami_only:
        for label, filename in types_to_eval.items():
            effective_k = _effective_topk(filename, topk)
            results[label] = evaluate_single(
                field_id, filename,
                gamma=args.gamma, n_runs=args.n_runs, topk=effective_k,
                node_subset=node_subset,
            )

    # ── 2. Combinations ──────────────────────────────────────────
    if not args.no_combo and not args.ami_only:
        type_files = list(types_to_eval.values())
        type_labels = list(types_to_eval.keys())

        # 2-way and 3-way combinations
        for r in range(2, len(type_files) + 1):
            for combo_idx in combinations(range(len(type_files)), r):
                combo_files = [type_files[i] for i in combo_idx]
                combo_labels = [type_labels[i] for i in combo_idx]
                combo_name_base = "+".join(combo_labels)

                for method in args.combo_methods:
                    name = f"{combo_name_base}_{method}"
                    results[name] = evaluate_combo(
                        field_id, combo_files, method,
                        gamma=args.gamma, n_runs=args.n_runs, topk=topk,
                        node_subset=node_subset,
                    )

    # ── 3. Print results ─────────────────────────────────────────
    if results:
        print_results(results, field_id)

    # ── 4. Cross-type AMI matrix ─────────────────────────────────
    if not args.no_ami:
        ami_types = ALL_LAYER_TYPES if args.all_layers else types_to_eval
        ami_df = compute_cross_ami(
            field_id, ami_types, gamma=args.gamma, topk=topk,
            node_subset=node_subset,
        )
        if ami_df.height > 0:
            print(f"\n{'='*80}")
            print(f"  CROSS-TYPE AMI MATRIX (seed=0, common nodes)")
            print(f"{'='*80}")
            print(ami_df)

    # ── 5. Save results ──────────────────────────────────────────
    parts = []
    if topk:
        parts.append(f"topk{topk}")
    if args.common_subset:
        parts.append(f"subset_{args.common_subset}")
    suffix = f"_{'_'.join(parts)}" if parts else ""
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    out_path = out_dir / f"eval_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", out_path)

    # Save AMI matrix
    if not args.no_ami and ami_df.height > 0:
        ami_df.write_parquet(out_dir / f"cross_type_ami{suffix}.parquet")
        log.info("AMI matrix saved to %s", out_dir / "cross_type_ami.parquet")


if __name__ == "__main__":
    main()
