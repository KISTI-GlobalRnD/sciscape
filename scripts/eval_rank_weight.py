#!/usr/bin/env python3
"""Compare raw cosine vs 1/rank weighting for embedding layers.

Converts per-node k-NN cosine similarities to 1/rank weights,
then evaluates clustering stability. This addresses the "weight
resolution" problem where cosine values are too uniform (~0.92±0.01).

Usage:
    .venv/bin/python scripts/eval_rank_weight.py --field 15 --n-runs 10
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
log = logging.getLogger("rank_w")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results_gcc"

LAYERS = {
    "Emb_full": "emb_full_knn30",
    "Emb_bg": "emb_bg_knn30",
    "Emb_nov": "emb_nov_knn30",
}

MIN_CLUSTER_SIZE = 5


def cosine_to_rank_weight(df: pl.DataFrame) -> pl.DataFrame:
    """Convert cosine similarity to 1/rank per node.

    For each node, rank neighbors by descending weight, then replace
    weight with 1/rank. This creates a natural weight gradient
    (1.0, 0.5, 0.33, 0.25, ...) regardless of original weight scale.
    """
    # Bidirectional: rank from both uid1 and uid2 perspective
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

    # Rank per node (descending weight → rank 1 = strongest)
    ranked = bidi.with_columns(
        pl.col("w")
        .rank("ordinal", descending=True)
        .over("node")
        .alias("rank")
    )

    # Weight = 1/rank
    ranked = ranked.with_columns(
        (1.0 / pl.col("rank")).alias("rank_weight"),
    )

    # Deduplicate to undirected: take max rank_weight for each pair
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


def build_graph(df: pl.DataFrame, topk: int | None = None) -> tuple[ig.Graph, list[str]]:
    """Build igraph from edge DataFrame, optionally top-k filtered."""
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

    if topk:
        M = _apply_topk(M, topk)

    upper = sparse.triu(M, k=1).tocoo()
    mask = upper.data > 0
    edges = list(zip(upper.row[mask].tolist(), upper.col[mask].tolist()))
    weights = upper.data[mask].astype(np.float64).tolist()

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    g = g.simplify(combine_edges="max")

    gcc_idx = g.connected_components().giant().vs.indices
    g_gcc = g.subgraph(gcc_idx)
    gcc_ids = [all_ids[i] for i in gcc_idx]
    return g_gcc, gcc_ids


def _apply_topk(M, k):
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
    M_f = sparse.csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(n, n))
    return M_f.maximum(M_f.T)


def auto_gamma(g):
    weights = np.array(g.es["weight"])
    n, m = g.vcount(), g.ecount()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    gamma = float(np.median(weights)) * density * 2.0
    return max(1e-8, min(gamma, 1.0))


def evaluate(g, gamma, n_runs):
    memberships, qualities, n_clusters = [], [], []
    for seed in range(n_runs):
        part = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma, weights="weight", seed=seed,
        )
        memberships.append(part.membership)
        qualities.append(part.quality())
        n_clusters.append(sum(1 for c in Counter(part.membership).values() if c >= MIN_CLUSTER_SIZE))

    # Stability
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

    # NMI
    sample = min(20, n_runs)
    rng = np.random.RandomState(42)
    idx_s = rng.choice(n_runs, sample, replace=False) if n_runs > sample else np.arange(n_runs)
    nmis = []
    for i in range(len(idx_s)):
        for j in range(i + 1, len(idx_s)):
            nmis.append(normalized_mutual_info_score(memberships[idx_s[i]], memberships[idx_s[j]]))

    weights = np.array(g.es["weight"])

    return {
        "n_nodes": n,
        "n_edges": g.ecount(),
        "gamma": gamma,
        "n_clusters_mean": float(np.mean(n_clusters)),
        "n_clusters_std": float(np.std(n_clusters)),
        "nmi_mean": float(np.mean(nmis)) if nmis else 0.0,
        "node_stability_mean": float(node_stab.mean()),
        "stable_pct": float(np.sum(node_stab > 0.9) / n),
        "unstable_pct": float(np.sum(node_stab < 0.5) / n),
        "weight_median": float(np.median(weights)),
        "weight_std": float(np.std(weights)),
        "weight_range": f"{weights.min():.4f}–{weights.max():.4f}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()

    field_id = args.field
    results = []

    for label, filename in LAYERS.items():
        path = EDGE_DIR / f"field_{field_id}" / f"{filename}.parquet"
        if not path.exists():
            log.warning("Missing %s", path)
            continue

        df_raw = pl.read_parquet(path)
        log.info("=== %s: %d raw edges ===", label, df_raw.height)

        # ── A. Raw cosine weights ──────────────────────────
        log.info("  [A] Raw cosine weights")
        t0 = time.time()
        g_raw, _ = build_graph(df_raw, topk=args.topk)
        gamma_raw = auto_gamma(g_raw)
        log.info("  Nodes=%d, Edges=%d, γ=%.2e", g_raw.vcount(), g_raw.ecount(), gamma_raw)
        r_raw = evaluate(g_raw, gamma_raw, args.n_runs)
        r_raw["label"] = f"{label}_cosine"
        r_raw["time_sec"] = time.time() - t0
        results.append(r_raw)

        # ── B. 1/rank weights ──────────────────────────────
        log.info("  [B] 1/rank weights")
        t0 = time.time()
        df_rank = cosine_to_rank_weight(df_raw)
        log.info("  Converted to %d rank-weighted edges", df_rank.height)
        g_rank, _ = build_graph(df_rank, topk=args.topk)
        gamma_rank = auto_gamma(g_rank)
        log.info("  Nodes=%d, Edges=%d, γ=%.2e", g_rank.vcount(), g_rank.ecount(), gamma_rank)
        r_rank = evaluate(g_rank, gamma_rank, args.n_runs)
        r_rank["label"] = f"{label}_1/rank"
        r_rank["time_sec"] = time.time() - t0
        results.append(r_rank)

    # Print comparison
    print(f"\n{'='*130}")
    print(f"  COSINE vs 1/RANK WEIGHTING — Field {field_id}")
    print(f"{'='*130}")
    print(f"{'Label':<20} {'Nodes':>8} {'Edges':>10} {'γ':>10} "
          f"{'#Clust':>8} {'NMI':>6} {'NodeStab':>8} {'Stable%':>8} {'Unstbl%':>8} "
          f"{'W_med':>8} {'W_std':>8} {'W_range':>20}")
    print("-" * 130)
    for r in results:
        print(f"{r['label']:<20} {r['n_nodes']:>8,} {r['n_edges']:>10,} "
              f"{r['gamma']:>10.2e} "
              f"{r['n_clusters_mean']:>7.0f}±{r['n_clusters_std']:<3.0f}"
              f"{r['nmi_mean']:>6.3f} "
              f"{r['node_stability_mean']:>8.3f} "
              f"{r['stable_pct']*100:>7.1f}% "
              f"{r['unstable_pct']*100:>7.1f}% "
              f"{r['weight_median']:>8.4f} {r['weight_std']:>8.4f} "
              f"{r['weight_range']:>20}")

    # Save
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_rank_weight.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
