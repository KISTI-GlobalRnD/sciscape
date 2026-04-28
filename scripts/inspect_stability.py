#!/usr/bin/env python3
"""Analyze WHY nodes are stable/unstable in each link-type layer.

For each layer, runs Leiden ensemble at matched γ, computes per-node
stability, then profiles stable vs unstable nodes by:
  - Degree, weight statistics
  - Cluster assignment entropy across runs
  - Boundary vs interior position
  - Cross-layer stability comparison (same node, different layers)
  - Metadata (year, citations, subfield)

Usage:
    .venv/bin/python scripts/inspect_stability.py --field 15 --n-runs 10
    .venv/bin/python scripts/inspect_stability.py --field 15 --n-runs 10 --fetch-titles
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import polars as pl
from scipy import sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stab_inspect")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results_gcc"
HDD = Path.home() / "Desktop/HDD/local_map_analysis_data/processed/outputs/data"
META_DIR = HDD / "oa26_gcc_only"

ALL_LAYERS = {
    "DC": "dc_fractional",
    "BC": "bc_assoc_strength",
    "CC": "cc_assoc_strength",
    "Emb_full": "emb_full_knn30",
    "Emb_bg": "emb_bg_knn30",
    "Emb_nov": "emb_nov_knn30",
}

# Matched γ from eval_gamma_sweep_k290.json (Field 15)
MATCHED_GAMMA = {
    15: {
        "DC": 1.6961e-06, "BC": 2.0951e-06, "CC": 1.8747e-05,
        "Emb_full": 7.9917e-03, "Emb_bg": 7.9045e-03, "Emb_nov": 7.9681e-03,
    },
}

MIN_CLUSTER_SIZE = 5


# ── Graph loading (reused from eval_gamma_sweep_gcc.py) ─────────

def apply_topk_backbone(M: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    n = M.shape[0]
    M_csr = M.tocsr()
    rows, cols, data = [], [], []
    for i in range(n):
        start, end = M_csr.indptr[i], M_csr.indptr[i + 1]
        if end - start <= k:
            cols_i = M_csr.indices[start:end]
            data_i = M_csr.data[start:end]
        else:
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
        (np.array(data), (np.array(rows), np.array(cols))), shape=(n, n),
    )
    return M_filtered.maximum(M_filtered.T)


def load_graph(field_id, link_type, topk=None):
    path = EDGE_DIR / f"field_{field_id}" / f"{link_type}.parquet"
    df = pl.read_parquet(path)
    all_ids = (
        pl.concat([df["uid1"].alias("id"), df["uid2"].alias("id")])
        .unique().sort().to_list()
    )
    id2idx = {wid: i for i, wid in enumerate(all_ids)}
    n = len(all_ids)

    src = np.array([id2idx[x] for x in df["uid1"].to_list()], dtype=np.int32)
    dst = np.array([id2idx[x] for x in df["uid2"].to_list()], dtype=np.int32)
    w = df["rel_sum2"].to_numpy().astype(np.float32)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    data = np.concatenate([w, w])
    M = sparse.csr_matrix((data, (row, col)), shape=(n, n))

    if topk is not None:
        M = apply_topk_backbone(M, topk)

    upper = sparse.triu(M, k=1).tocoo()
    mask = upper.data > 0
    edges = list(zip(upper.row[mask].tolist(), upper.col[mask].tolist()))
    weights = upper.data[mask].astype(np.float64).tolist()

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    g = g.simplify(combine_edges="max")

    comps = g.connected_components()
    gcc_idx = comps.giant().vs.indices
    g_gcc = g.subgraph(gcc_idx)
    gcc_ids = [all_ids[i] for i in gcc_idx]
    return g_gcc, gcc_ids


# ── Per-node stability computation ──────────────────────────────

def compute_node_stability(memberships: list[list[int]], g: ig.Graph) -> np.ndarray:
    """Compute per-node stability score."""
    n = g.vcount()
    mem_arr = np.array(memberships)
    edges_arr = np.array(g.get_edgelist())
    src, dst = edges_arr[:, 0], edges_arr[:, 1]
    co_rates = np.mean(mem_arr[:, src] == mem_arr[:, dst], axis=0)

    node_stab = np.zeros(n)
    node_deg = np.zeros(n)
    np.add.at(node_stab, src, co_rates)
    np.add.at(node_stab, dst, co_rates)
    np.add.at(node_deg, src, 1.0)
    np.add.at(node_deg, dst, 1.0)
    mask = node_deg > 0
    node_stab[mask] /= node_deg[mask]
    node_stab[~mask] = 1.0
    return node_stab


def compute_assignment_entropy(memberships: list[list[int]], n: int) -> np.ndarray:
    """Compute entropy of cluster assignments per node across runs."""
    mem_arr = np.array(memberships)  # (n_runs, n_nodes)
    n_runs = mem_arr.shape[0]
    entropy = np.zeros(n)
    for i in range(n):
        counts = Counter(mem_arr[:, i].tolist())
        probs = np.array(list(counts.values())) / n_runs
        entropy[i] = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy


def compute_boundary_score(memberships: list[list[int]], g: ig.Graph) -> np.ndarray:
    """For each node, fraction of neighbors in a different cluster (avg across runs)."""
    n = g.vcount()
    mem_arr = np.array(memberships)
    adj = g.get_adjacency_sparse()
    adj_csr = adj.tocsr()

    boundary = np.zeros(n)
    n_runs = mem_arr.shape[0]

    for run_idx in range(n_runs):
        mem = mem_arr[run_idx]
        for i in range(n):
            start, end = adj_csr.indptr[i], adj_csr.indptr[i + 1]
            if end == start:
                continue
            nbr_idx = adj_csr.indices[start:end]
            diff_frac = np.mean(mem[nbr_idx] != mem[i])
            boundary[i] += diff_frac

    boundary /= n_runs
    return boundary


def compute_weight_stats(g: ig.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-node: mean weight, max weight, weight CV."""
    n = g.vcount()
    mean_w = np.zeros(n)
    max_w = np.zeros(n)
    cv_w = np.zeros(n)
    degrees = np.array(g.degree())

    adj = g.get_adjacency_sparse()
    adj_csr = adj.tocsr()
    weights = np.array(g.es["weight"])

    # Build weighted adjacency
    edges = np.array(g.get_edgelist())
    src, dst = edges[:, 0], edges[:, 1]
    W = sparse.csr_matrix((weights, (src, dst)), shape=(n, n))
    W = W + W.T  # symmetrize

    for i in range(n):
        start, end = W.indptr[i], W.indptr[i + 1]
        if end == start:
            continue
        ws = W.data[start:end]
        mean_w[i] = ws.mean()
        max_w[i] = ws.max()
        cv_w[i] = ws.std() / (ws.mean() + 1e-12)

    return mean_w, max_w, cv_w


# ── Profile nodes ───────────────────────────────────────────────

def profile_layer(
    field_id: int,
    label: str,
    filename: str,
    gamma: float,
    n_runs: int,
    topk: int = 30,
) -> dict:
    """Run ensemble and compute per-node profiles for one layer."""
    log.info("=== Profiling %s (γ=%.2e) ===", label, gamma)
    effective_k = topk if "dc_" not in filename else None
    g, node_ids = load_graph(field_id, filename, topk=effective_k)
    log.info("  %d nodes, %d edges", g.vcount(), g.ecount())

    # Ensemble
    memberships = []
    for seed in range(n_runs):
        part = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma, weights="weight", seed=seed,
        )
        memberships.append(part.membership)

    # Per-node metrics
    node_stab = compute_node_stability(memberships, g)
    entropy = compute_assignment_entropy(memberships, g.vcount())
    boundary = compute_boundary_score(memberships, g)
    mean_w, max_w, cv_w = compute_weight_stats(g)
    degrees = np.array(g.degree())

    # Cluster size for seed=0
    mem0 = memberships[0]
    cluster_sizes = Counter(mem0)

    return {
        "label": label,
        "graph": g,
        "node_ids": node_ids,
        "memberships": memberships,
        "node_stab": node_stab,
        "entropy": entropy,
        "boundary": boundary,
        "degrees": degrees,
        "mean_w": mean_w,
        "max_w": max_w,
        "cv_w": cv_w,
        "cluster_sizes": cluster_sizes,
        "mem0": mem0,
    }


def print_stability_profile(profiles: dict[str, dict], field_id: int):
    """Print aggregate statistics comparing stable vs unstable nodes."""
    print(f"\n{'='*130}")
    print(f"  NODE STABILITY PROFILE — Field {field_id}")
    print(f"{'='*130}")

    for label, p in profiles.items():
        stab = p["node_stab"]
        deg = p["degrees"]
        ent = p["entropy"]
        bnd = p["boundary"]
        mw = p["mean_w"]
        cv = p["cv_w"]

        stable_mask = stab > 0.9
        unstable_mask = stab < 0.5
        mid_mask = (stab >= 0.5) & (stab <= 0.9)

        n = len(stab)
        print(f"\n── {label} ({n:,} nodes) ──")
        print(f"  {'Metric':<25} {'Stable(>0.9)':>14} {'Mid(0.5-0.9)':>14} {'Unstable(<0.5)':>14}")
        print(f"  {'-'*70}")
        print(f"  {'Count':<25} {stable_mask.sum():>14,} {mid_mask.sum():>14,} {unstable_mask.sum():>14,}")
        print(f"  {'Fraction':<25} {stable_mask.mean()*100:>13.1f}% {mid_mask.mean()*100:>13.1f}% {unstable_mask.mean()*100:>13.1f}%")
        print(f"  {'Degree (mean)':<25} {deg[stable_mask].mean():>14.1f} {deg[mid_mask].mean():>14.1f} {deg[unstable_mask].mean() if unstable_mask.any() else 0:>14.1f}")
        print(f"  {'Degree (median)':<25} {np.median(deg[stable_mask]):>14.0f} {np.median(deg[mid_mask]):>14.0f} {np.median(deg[unstable_mask]) if unstable_mask.any() else 0:>14.0f}")
        print(f"  {'Mean weight':<25} {mw[stable_mask].mean():>14.6f} {mw[mid_mask].mean():>14.6f} {mw[unstable_mask].mean() if unstable_mask.any() else 0:>14.6f}")
        print(f"  {'Weight CV':<25} {cv[stable_mask].mean():>14.3f} {cv[mid_mask].mean():>14.3f} {cv[unstable_mask].mean() if unstable_mask.any() else 0:>14.3f}")
        print(f"  {'Entropy (bits)':<25} {ent[stable_mask].mean():>14.3f} {ent[mid_mask].mean():>14.3f} {ent[unstable_mask].mean() if unstable_mask.any() else 0:>14.3f}")
        print(f"  {'Boundary frac':<25} {bnd[stable_mask].mean():>14.3f} {bnd[mid_mask].mean():>14.3f} {bnd[unstable_mask].mean() if unstable_mask.any() else 0:>14.3f}")

        # Cluster size analysis
        mem0 = np.array(p["mem0"])
        csizes = p["cluster_sizes"]
        node_csize = np.array([csizes[c] for c in mem0])
        print(f"  {'Cluster size (mean)':<25} {node_csize[stable_mask].mean():>14.0f} {node_csize[mid_mask].mean():>14.0f} {node_csize[unstable_mask].mean() if unstable_mask.any() else 0:>14.0f}")


def print_cross_layer_comparison(profiles: dict[str, dict], field_id: int):
    """Show same nodes' stability across different layers."""
    print(f"\n{'='*130}")
    print(f"  CROSS-LAYER STABILITY — Field {field_id}")
    print(f"{'='*130}")

    # Find common nodes
    all_id_sets = {label: set(p["node_ids"]) for label, p in profiles.items()}
    common = set.intersection(*all_id_sets.values())
    log.info("Common nodes across all layers: %d", len(common))

    # Build id→stab mapping per layer
    stab_maps = {}
    for label, p in profiles.items():
        stab_maps[label] = {nid: p["node_stab"][i] for i, nid in enumerate(p["node_ids"]) if nid in common}

    common_list = sorted(common)

    # Correlation between layers
    from scipy.stats import spearmanr
    labels = list(profiles.keys())
    print(f"\n  Spearman correlation of per-node stability:")
    print(f"  {'':>12}", end="")
    for b in labels:
        print(f" {b:>10}", end="")
    print()
    for a in labels:
        print(f"  {a:>12}", end="")
        for b in labels:
            sa = np.array([stab_maps[a][n] for n in common_list])
            sb = np.array([stab_maps[b][n] for n in common_list])
            rho, _ = spearmanr(sa, sb)
            print(f" {rho:>10.3f}", end="")
        print()

    # Nodes stable everywhere vs unstable everywhere
    n_stable_all = 0
    n_unstable_all = 0
    n_mixed = 0
    for nid in common_list:
        stabs = [stab_maps[label][nid] for label in labels]
        if all(s > 0.9 for s in stabs):
            n_stable_all += 1
        elif all(s < 0.5 for s in stabs):
            n_unstable_all += 1
        else:
            n_mixed += 1
    total = len(common_list)
    print(f"\n  Stable in ALL layers (>0.9): {n_stable_all:,} ({n_stable_all/total*100:.1f}%)")
    print(f"  Unstable in ALL layers (<0.5): {n_unstable_all:,} ({n_unstable_all/total*100:.1f}%)")
    print(f"  Mixed: {n_mixed:,} ({n_mixed/total*100:.1f}%)")

    # Layer-specific instability: unstable in layer X but stable in most others
    print(f"\n  Layer-specific instability (unstable in X, stable in ≥3 others):")
    for target_label in labels:
        count = 0
        for nid in common_list:
            if stab_maps[target_label][nid] < 0.5:
                stable_others = sum(1 for l in labels if l != target_label and stab_maps[l][nid] > 0.9)
                if stable_others >= 3:
                    count += 1
        print(f"    {target_label}: {count:,} nodes")


def print_sample_nodes(
    profiles: dict[str, dict],
    meta: pl.DataFrame | None,
    field_id: int,
    n_sample: int = 5,
    fetch_titles: bool = False,
):
    """Sample and print specific stable/unstable nodes with details."""
    print(f"\n{'='*130}")
    print(f"  SAMPLE NODE ANALYSIS — Field {field_id}")
    print(f"{'='*130}")

    # Build meta lookup
    meta_dict = {}
    if meta is not None:
        for row in meta.iter_rows(named=True):
            meta_dict[row["work_id"]] = row

    for label, p in profiles.items():
        stab = p["node_stab"]
        deg = p["degrees"]
        ent = p["entropy"]
        bnd = p["boundary"]
        ids = p["node_ids"]

        # Sample: n_sample most unstable, n_sample most stable
        unstable_idx = np.argsort(stab)[:n_sample]
        stable_idx = np.argsort(stab)[-n_sample:][::-1]

        print(f"\n── {label}: Most STABLE nodes ──")
        _print_node_details(stable_idx, p, meta_dict, fetch_titles)

        print(f"\n── {label}: Most UNSTABLE nodes ──")
        _print_node_details(unstable_idx, p, meta_dict, fetch_titles)


def _print_node_details(indices, profile, meta_dict, fetch_titles):
    """Print detailed info for selected node indices."""
    stab = profile["node_stab"]
    deg = profile["degrees"]
    ent = profile["entropy"]
    bnd = profile["boundary"]
    ids = profile["node_ids"]
    mem_arr = np.array(profile["memberships"])
    csizes = profile["cluster_sizes"]
    mem0 = profile["mem0"]

    for idx in indices:
        nid = ids[idx]
        m = meta_dict.get(nid, {})
        title = ""
        if fetch_titles:
            title = _fetch_title(nid)

        # Cluster assignments across runs
        assignments = mem_arr[:, idx]
        assign_counts = Counter(assignments.tolist())
        top_clusters = assign_counts.most_common(3)
        assign_str = ", ".join(f"c{c}({cnt}x)" for c, cnt in top_clusters)

        csize = csizes.get(mem0[idx], 0)

        print(f"  {nid} | stab={stab[idx]:.3f} | deg={deg[idx]} | entropy={ent[idx]:.2f}bits"
              f" | boundary={bnd[idx]:.2f} | csize={csize}")
        if title:
            print(f"    title: {title}")
        print(f"    year={m.get('publication_year','?')} | cited={m.get('cited_by_count','?')}"
              f" | subfield={m.get('subfield_name','?')}")
        print(f"    assignments: {assign_str}")


def _fetch_title(work_id: str) -> str:
    import urllib.request
    oa_id = work_id.replace("W", "")
    url = f"https://api.openalex.org/works/W{oa_id}?select=title"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("title", "N/A")
    except Exception:
        return "N/A"


def main():
    parser = argparse.ArgumentParser(description="Analyze node stability per layer")
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n-sample", type=int, default=5,
                        help="Number of stable/unstable nodes to sample per layer")
    parser.add_argument("--fetch-titles", action="store_true")
    parser.add_argument("--layers", nargs="*", default=None,
                        help="Specific layers (default: all 6)")
    args = parser.parse_args()

    field_id = args.field

    # Load matched gammas
    gamma_path = OUT_DIR / f"field_{field_id}" / f"eval_gamma_sweep_k290.json"
    if gamma_path.exists():
        with open(gamma_path) as f:
            sweep_results = json.load(f)
        gammas = {r["label"]: r["gamma_matched"] for r in sweep_results}
        log.info("Loaded matched gammas from %s", gamma_path)
    elif field_id in MATCHED_GAMMA:
        gammas = MATCHED_GAMMA[field_id]
    else:
        log.error("No matched gammas for field %d", field_id)
        return

    # Determine layers
    if args.layers:
        layers = {k: v for k, v in ALL_LAYERS.items() if k in args.layers}
    else:
        layers = dict(ALL_LAYERS)

    # Load metadata
    meta = None
    meta_path = META_DIR / f"field_{field_id}_nodes_oa_meta_gcc.parquet"
    if meta_path.exists():
        meta = pl.read_parquet(meta_path)
        log.info("Loaded metadata: %d nodes", meta.height)

    # Profile each layer
    profiles = {}
    for label, filename in layers.items():
        if label not in gammas:
            log.warning("No gamma for %s, skipping", label)
            continue
        path = EDGE_DIR / f"field_{field_id}" / f"{filename}.parquet"
        if not path.exists():
            log.warning("Missing %s, skipping", path)
            continue
        profiles[label] = profile_layer(
            field_id, label, filename, gammas[label], args.n_runs, args.topk,
        )

    # Print results
    print_stability_profile(profiles, field_id)
    print_cross_layer_comparison(profiles, field_id)
    print_sample_nodes(profiles, meta, field_id, args.n_sample, args.fetch_titles)

    # Save per-node stability for further analysis
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save common-node stability matrix
    all_id_sets = {label: set(p["node_ids"]) for label, p in profiles.items()}
    common = sorted(set.intersection(*all_id_sets.values()))
    rows = []
    for nid in common:
        row = {"work_id": nid}
        for label, p in profiles.items():
            idx = p["node_ids"].index(nid) if nid in set(p["node_ids"]) else -1
            if idx >= 0:
                row[f"stab_{label}"] = float(p["node_stab"][idx])
                row[f"deg_{label}"] = int(p["degrees"][idx])
                row[f"entropy_{label}"] = float(p["entropy"][idx])
                row[f"boundary_{label}"] = float(p["boundary"][idx])
        rows.append(row)

    df_out = pl.DataFrame(rows)
    out_path = out_dir / "node_stability_profiles.parquet"
    df_out.write_parquet(out_path)
    log.info("Saved %d node profiles to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
