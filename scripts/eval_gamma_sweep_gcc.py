#!/usr/bin/env python3
"""γ sweep to match k_eff across link types for fair comparison.

Uses binary search to find γ that produces a target cluster count
(reference: BC top-k30 auto_gamma), then evaluates each layer at
that matched γ.

Usage:
    .venv/bin/python scripts/eval_gamma_sweep_gcc.py --field 15 --n-runs 10
    .venv/bin/python scripts/eval_gamma_sweep_gcc.py --field 12 --n-runs 10
    .venv/bin/python scripts/eval_gamma_sweep_gcc.py --field 15 --target-k 200
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
from sklearn.metrics import normalized_mutual_info_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gamma_sweep")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results_gcc"

ALL_LAYERS = {
    "DC": "dc_fractional",
    "BC": "bc_assoc_strength",
    "CC": "cc_assoc_strength",
    "Emb_full": "emb_full_knn30",
    "Emb_bg": "emb_bg_knn30",
    "Emb_nov": "emb_nov_knn30",
}

MIN_CLUSTER_SIZE = 5


# ── Reused from eval_linktype_gcc.py ────────────────────────────

def apply_topk_backbone(M: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Keep only top-k edges per node by weight (symmetric)."""
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


def _effective_topk(filename: str, topk: int | None) -> int | None:
    if topk is None:
        return None
    if "dc_" in filename:
        log.info("  Skipping top-k for DC (uniform weights)")
        return None
    return topk


def _cosine_to_rank_weight(df: pl.DataFrame) -> pl.DataFrame:
    """Convert cosine similarity to 1/rank per node."""
    fwd = df.select(
        pl.col("uid1").alias("node"),
        pl.col("uid2").alias("neighbor"),
        pl.col("rel_sum2").alias("w"),
    )
    rev = df.select(
        pl.col("uid2").alias("node"),
        pl.col("uid1").alias("neighbor"),
        pl.col("rel_sum2").alias("w"),
    )
    bidi = pl.concat([fwd, rev])
    ranked = bidi.with_columns(
        pl.col("w").rank("ordinal", descending=True).over("node").alias("rank")
    )
    ranked = ranked.with_columns((1.0 / pl.col("rank")).alias("rank_weight"))
    edges = (
        ranked
        .with_columns(
            pl.min_horizontal("node", "neighbor").alias("lo"),
            pl.max_horizontal("node", "neighbor").alias("hi"),
        )
        .group_by("lo", "hi")
        .agg(pl.col("rank_weight").max())
        .rename({"lo": "uid1", "hi": "uid2", "rank_weight": "rel_sum2"})
    )
    return edges


def load_graph(
    field_id: int, link_type: str, topk: int | None = None,
    rank_weight: bool = False,
) -> tuple[ig.Graph, list[str]]:
    """Load edge list → igraph (GCC)."""
    path = EDGE_DIR / f"field_{field_id}" / f"{link_type}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pl.read_parquet(path)

    # Apply 1/rank weighting for embedding layers
    if rank_weight and "emb" in link_type:
        log.info("    applying 1/rank weight transform")
        df = _cosine_to_rank_weight(df)

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
        before = M.nnz // 2
        M = apply_topk_backbone(M, topk)
        after = M.nnz // 2
        log.info("    top-k(%d): %d → %d edges (%.1f%%)", topk, before, after, after / before * 100)

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
    log.info("  %s: %d nodes, %d edges → GCC: %d, %d",
             link_type, n, len(edges), g_gcc.vcount(), g_gcc.ecount())
    return g_gcc, gcc_ids


def auto_gamma(g: ig.Graph) -> float:
    weights = np.array(g.es["weight"])
    n, m = g.vcount(), g.ecount()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    gamma = float(np.median(weights)) * density * 2.0
    gamma = max(gamma, 1e-8)
    gamma = min(gamma, 1.0)
    return gamma


def count_clusters(g: ig.Graph, gamma: float, seed: int = 0) -> int:
    """Count large clusters (≥ MIN_CLUSTER_SIZE) at given γ."""
    part = leidenalg.find_partition(
        g, leidenalg.CPMVertexPartition,
        resolution_parameter=gamma, weights="weight", seed=seed,
    )
    return sum(1 for c in Counter(part.membership).values() if c >= MIN_CLUSTER_SIZE)


# ── Search for target cluster count ──────────────────────────────

def search_gamma(
    g: ig.Graph,
    target_k: int,
    tol: float = 0.1,
    max_iter: int = 15,
    seed: int = 0,
) -> tuple[float, int]:
    """Fast γ search: sparse probe → bracket → geometric bisection.

    Minimizes Leiden calls (~8-12 total instead of 25+).
    """
    _cache: dict[float, int] = {}

    def count_k(gamma):
        gamma = round(gamma, 12)
        if gamma in _cache:
            return _cache[gamma]
        k = count_clusters(g, gamma, seed)
        _cache[gamma] = k
        return k

    min_k = int(target_k * (1 - tol))
    max_k = int(target_k * (1 + tol))

    ag = auto_gamma(g)
    best_gamma = ag
    best_k = count_k(ag)
    best_dist = abs(best_k - target_k)

    log.info("    target: %d clusters (range %d–%d)", target_k, min_k, max_k)
    log.info("    auto_gamma=%.2e → k=%d", ag, best_k)

    if min_k <= best_k <= max_k:
        log.info("    auto_gamma already in range!")
        return best_gamma, best_k

    # Phase 1: Sparse probe (5 points) to find bracket
    results = [(ag, best_k)]
    for exp in [-2, -1, 0, 1, 2]:
        gi = ag * (10 ** exp)
        if 1e-10 < gi < 10.0:
            ki = count_k(gi)
            results.append((gi, ki))
            d = abs(ki - target_k)
            if d < best_dist:
                best_gamma, best_k, best_dist = gi, ki, d
            if min_k <= ki <= max_k:
                log.info("    probe hit: γ=%.2e → k=%d ✓", gi, ki)
                return gi, ki

    # Phase 2: Find bracket
    results.sort(key=lambda x: x[0])
    lo_g, hi_g, lo_k, hi_k = None, None, None, None
    for i in range(len(results) - 1):
        g1, k1 = results[i]
        g2, k2 = results[i + 1]
        if (k1 <= target_k <= k2) or (k2 <= target_k <= k1):
            lo_g, hi_g, lo_k, hi_k = g1, g2, k1, k2
            break

    if lo_g is None:
        # Fill gaps
        for exp in [-1.5, 0.5, -0.5, 1.5]:
            gi = ag * (10 ** exp)
            if 1e-10 < gi < 10.0:
                ki = count_k(gi)
                results.append((gi, ki))
                d = abs(ki - target_k)
                if d < best_dist:
                    best_gamma, best_k, best_dist = gi, ki, d
                if min_k <= ki <= max_k:
                    return gi, ki
        results.sort(key=lambda x: x[0])
        for i in range(len(results) - 1):
            g1, k1 = results[i]
            g2, k2 = results[i + 1]
            if (k1 <= target_k <= k2) or (k2 <= target_k <= k1):
                lo_g, hi_g, lo_k, hi_k = g1, g2, k1, k2
                break

    if lo_g is None:
        log.info("    no bracket found, best: γ=%.2e → k=%d", best_gamma, best_k)
        return best_gamma, best_k

    log.info("    bracket: [%.2e(k=%d), %.2e(k=%d)]", lo_g, lo_k, hi_g, hi_k)

    # Phase 3: Geometric bisection
    for i in range(max_iter):
        mid = np.sqrt(lo_g * hi_g)
        km = count_k(mid)
        d = abs(km - target_k)
        if d < best_dist:
            best_gamma, best_k, best_dist = mid, km, d
        if min_k <= km <= max_k:
            log.info("    bisect %d: γ=%.2e → k=%d ✓", i, mid, km)
            return mid, km

        if (lo_k < target_k and km < target_k) or (lo_k > target_k and km > target_k):
            lo_g, lo_k = mid, km
        else:
            hi_g, hi_k = mid, km

        if abs(hi_g / lo_g - 1) < 1e-4:
            break

    log.info("    converged: γ=%.2e → k=%d (target=%d)", best_gamma, best_k, target_k)
    return best_gamma, best_k


# ── Ensemble evaluation ─────────────────────────────────────────

def leiden_ensemble(g: ig.Graph, gamma: float, n_runs: int):
    memberships, qualities, n_clusters = [], [], []
    for seed in range(n_runs):
        part = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma, weights="weight", seed=seed,
        )
        memberships.append(part.membership)
        qualities.append(part.quality())
        n_clusters.append(sum(1 for c in Counter(part.membership).values() if c >= MIN_CLUSTER_SIZE))
    return memberships, qualities, n_clusters


def compute_stability(memberships, g):
    n = g.vcount()
    n_runs = len(memberships)
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

    degrees = np.array(g.degree())
    top10 = max(1, int(n * 0.1))
    hub_idx = np.argsort(degrees)[-top10:]

    sample = min(20, n_runs)
    rng = np.random.RandomState(42)
    idx_s = rng.choice(n_runs, sample, replace=False) if n_runs > sample else np.arange(n_runs)
    nmis = []
    for i in range(len(idx_s)):
        for j in range(i + 1, len(idx_s)):
            nmis.append(normalized_mutual_info_score(
                memberships[idx_s[i]], memberships[idx_s[j]],
            ))
    nmis = np.array(nmis)

    return {
        "node_stability_mean": float(node_stab.mean()),
        "stable_pct": float(np.sum(node_stab > 0.9) / n),
        "unstable_pct": float(np.sum(node_stab < 0.5) / n),
        "hub_stability": float(node_stab[hub_idx].mean()),
        "edge_co_mean": float(co_rates.mean()),
        "nmi_mean": float(nmis.mean()) if len(nmis) > 0 else 0.0,
        "nmi_std": float(nmis.std()) if len(nmis) > 0 else 0.0,
    }


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="γ sweep for k_eff matching")
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--target-k", type=int, default=None,
                        help="Target cluster count (default: use BC auto_gamma result)")
    parser.add_argument("--tol", type=float, default=0.1,
                        help="Tolerance for target matching (default: 0.1 = ±10%%)")
    parser.add_argument("--layers", nargs="*", default=None,
                        help="Specific layers to evaluate (default: all 6)")
    parser.add_argument("--rank-weight", action="store_true",
                        help="Use 1/rank weighting for embedding layers")
    args = parser.parse_args()

    field_id = args.field
    topk = args.topk

    # Determine layers
    if args.layers:
        layers = {k: v for k, v in ALL_LAYERS.items() if k in args.layers}
    else:
        layers = dict(ALL_LAYERS)

    rank_w = args.rank_weight
    if rank_w:
        log.info("Using 1/rank weighting for embedding layers")

    # ── Step 1: Determine reference k_eff from BC ────────────────
    log.info("=== Step 1: Reference k_eff from BC ===")
    g_bc, _ = load_graph(field_id, "bc_assoc_strength", topk=topk)
    gamma_bc = auto_gamma(g_bc)
    ref_k = count_clusters(g_bc, gamma_bc)
    log.info("  BC auto_gamma=%.2e → k_eff=%d", gamma_bc, ref_k)

    target_k = args.target_k or ref_k
    log.info("  Using target_k=%d (tol=%.0f%%)", target_k, args.tol * 100)

    # ── Step 2: Binary search γ for each layer ───────────────────
    log.info("=== Step 2: γ search per layer ===")
    matched = {}
    for label, filename in layers.items():
        log.info("── %s ──", label)
        t0 = time.time()
        effective_k = _effective_topk(filename, topk)
        g, node_ids = load_graph(field_id, filename, topk=effective_k, rank_weight=rank_w)
        gamma_matched, k_achieved = search_gamma(g, target_k, tol=args.tol)
        matched[label] = {
            "graph": g,
            "node_ids": node_ids,
            "gamma": gamma_matched,
            "k_achieved": k_achieved,
            "auto_gamma": auto_gamma(g),
            "search_sec": time.time() - t0,
        }
        log.info("  %s: γ=%.2e → k=%d (target=%d, auto_γ=%.2e)",
                 label, gamma_matched, k_achieved, target_k, matched[label]["auto_gamma"])

    # ── Step 3: Ensemble evaluation at matched γ ─────────────────
    log.info("=== Step 3: Ensemble evaluation ===")
    results = []
    for label, info in matched.items():
        g = info["graph"]
        gamma = info["gamma"]
        log.info("── %s: γ=%.2e (n_runs=%d) ──", label, gamma, args.n_runs)

        t0 = time.time()
        memberships, qualities, n_clusters = leiden_ensemble(g, gamma, args.n_runs)
        stab = compute_stability(memberships, g)
        eval_time = time.time() - t0

        results.append({
            "label": label,
            "n_nodes": g.vcount(),
            "n_edges": g.ecount(),
            "gamma_matched": gamma,
            "gamma_auto": info["auto_gamma"],
            "target_k": target_k,
            "k_achieved": info["k_achieved"],
            "n_clusters_mean": float(np.mean(n_clusters)),
            "n_clusters_std": float(np.std(n_clusters)),
            **stab,
            "quality_mean": float(np.mean(qualities)),
            "search_sec": info["search_sec"],
            "eval_sec": eval_time,
        })

    # ── Step 4: Print comparison table ───────────────────────────
    print(f"\n{'='*130}")
    print(f"  γ-MATCHED EVALUATION — Field {field_id} | target_k={target_k} (±{args.tol*100:.0f}%)")
    print(f"{'='*130}")
    print(f"{'Layer':<12} {'Nodes':>8} {'Edges':>10} "
          f"{'γ_auto':>10} {'γ_matched':>10} "
          f"{'k_eff':>8} {'#Clust':>8} {'NMI':>6} "
          f"{'NodeStab':>8} {'HubStab':>8} {'Stable%':>8} {'Unstbl%':>8}")
    print("-" * 130)
    for r in results:
        print(f"{r['label']:<12} {r['n_nodes']:>8,} {r['n_edges']:>10,} "
              f"{r['gamma_auto']:>10.2e} {r['gamma_matched']:>10.2e} "
              f"{r['k_achieved']:>8} "
              f"{r['n_clusters_mean']:>7.0f}±{r['n_clusters_std']:<3.0f}"
              f"{r['nmi_mean']:>6.3f} "
              f"{r['node_stability_mean']:>8.3f} {r['hub_stability']:>8.3f} "
              f"{r['stable_pct']*100:>7.1f}% "
              f"{r['unstable_pct']*100:>7.1f}%")

    # ── Step 5: Save ─────────────────────────────────────────────
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_rankw" if rank_w else ""
    out_path = out_dir / f"eval_gamma_sweep_k{target_k}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
