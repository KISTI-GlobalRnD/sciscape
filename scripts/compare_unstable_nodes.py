#!/usr/bin/env python3
"""Compare which nodes are unstable: cosine vs 1/rank, at matched k_eff.

Quick check: do the same nodes become unstable, or different ones?
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import polars as pl
from scipy import sparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("cmp")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"


def apply_topk(M, k):
    n = M.shape[0]
    M_csr = M.tocsr()
    rows, cols, data = [], [], []
    for i in range(n):
        s, e = M_csr.indptr[i], M_csr.indptr[i + 1]
        if e - s <= k:
            cols_i, data_i = M_csr.indices[s:e], M_csr.data[s:e]
        else:
            ld, lc = M_csr.data[s:e], M_csr.indices[s:e]
            idx = np.argpartition(ld, -k)[-k:]
            cols_i, data_i = lc[idx], ld[idx]
        for c, w in zip(cols_i, data_i):
            rows.append(i); cols.append(c); data.append(w)
    M_f = sparse.csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(n, n))
    return M_f.maximum(M_f.T)


def cosine_to_rank(df):
    fwd = df.select(pl.col("uid1").alias("node"), pl.col("uid2").alias("neighbor"), pl.col("rel_sum2").alias("w"))
    rev = df.select(pl.col("uid2").alias("node"), pl.col("uid1").alias("neighbor"), pl.col("rel_sum2").alias("w"))
    bidi = pl.concat([fwd, rev])
    ranked = bidi.with_columns(pl.col("w").rank("ordinal", descending=True).over("node").alias("rank"))
    ranked = ranked.with_columns((1.0 / pl.col("rank")).alias("rw"))
    return (
        ranked.with_columns(pl.min_horizontal("node", "neighbor").alias("lo"), pl.max_horizontal("node", "neighbor").alias("hi"))
        .group_by("lo", "hi").agg(pl.col("rw").max())
        .rename({"lo": "uid1", "hi": "uid2", "rw": "rel_sum2"})
    )


def build_graph(df, topk=30):
    all_ids = pl.concat([df["uid1"].alias("id"), df["uid2"].alias("id")]).unique().sort().to_list()
    id2idx = {w: i for i, w in enumerate(all_ids)}
    n = len(all_ids)
    src = np.array([id2idx[x] for x in df["uid1"].to_list()], dtype=np.int32)
    dst = np.array([id2idx[x] for x in df["uid2"].to_list()], dtype=np.int32)
    w = df["rel_sum2"].to_numpy().astype(np.float32)
    M = sparse.csr_matrix((np.concatenate([w,w]), (np.concatenate([src,dst]), np.concatenate([dst,src]))), shape=(n,n))
    if topk:
        M = apply_topk(M, topk)
    upper = sparse.triu(M, k=1).tocoo()
    mask = upper.data > 0
    g = ig.Graph(n=n, edges=list(zip(upper.row[mask].tolist(), upper.col[mask].tolist())), directed=False)
    g.es["weight"] = upper.data[mask].astype(np.float64).tolist()
    g = g.simplify(combine_edges="max")
    gcc_idx = g.connected_components().giant().vs.indices
    return g.subgraph(gcc_idx), [all_ids[i] for i in gcc_idx]


def node_stability(memberships, g):
    n = g.vcount()
    mem = np.array(memberships)
    edges = np.array(g.get_edgelist())
    s, d = edges[:,0], edges[:,1]
    co = np.mean(mem[:,s] == mem[:,d], axis=0)
    ns = np.zeros(n); nd = np.zeros(n)
    np.add.at(ns, s, co); np.add.at(ns, d, co)
    np.add.at(nd, s, 1.0); np.add.at(nd, d, 1.0)
    m = nd > 0; ns[m] /= nd[m]; ns[~m] = 1.0
    return ns


def run_ensemble(g, gamma, n_runs=10):
    mems = []
    for seed in range(n_runs):
        p = leidenalg.find_partition(g, leidenalg.CPMVertexPartition, resolution_parameter=gamma, weights="weight", seed=seed)
        mems.append(p.membership)
    return mems


def main():
    field_id = 15
    n_runs = 10
    # Matched gammas from sweep results
    gamma_cosine = 7.9917e-03   # Emb_full cosine matched
    gamma_rank = 8.07e-04       # Emb_full 1/rank matched

    df_raw = pl.read_parquet(EDGE_DIR / f"field_{field_id}" / "emb_full_knn30.parquet")
    log.info("Raw edges: %d", df_raw.height)

    # ── Cosine ──
    log.info("=== Cosine (γ=%.2e) ===", gamma_cosine)
    g_cos, ids_cos = build_graph(df_raw, topk=30)
    mems_cos = run_ensemble(g_cos, gamma_cosine, n_runs)
    stab_cos = node_stability(mems_cos, g_cos)

    # ── 1/rank ──
    log.info("=== 1/rank (γ=%.2e) ===", gamma_rank)
    df_rank = cosine_to_rank(df_raw)
    g_rank, ids_rank = build_graph(df_rank, topk=30)
    mems_rank = run_ensemble(g_rank, gamma_rank, n_runs)
    stab_rank = node_stability(mems_rank, g_rank)

    # ── Compare on common nodes ──
    set_cos, set_rank = set(ids_cos), set(ids_rank)
    common = sorted(set_cos & set_rank)
    log.info("Common nodes: %d", len(common))

    idx_cos = {nid: i for i, nid in enumerate(ids_cos)}
    idx_rank = {nid: i for i, nid in enumerate(ids_rank)}

    sc = np.array([stab_cos[idx_cos[n]] for n in common])
    sr = np.array([stab_rank[idx_rank[n]] for n in common])

    # Correlation
    from scipy.stats import spearmanr, pearsonr
    rho_s, _ = spearmanr(sc, sr)
    rho_p, _ = pearsonr(sc, sr)

    unstable_cos = set(n for n, s in zip(common, sc) if s < 0.5)
    unstable_rank = set(n for n, s in zip(common, sr) if s < 0.5)
    stable_cos = set(n for n, s in zip(common, sc) if s > 0.9)
    stable_rank = set(n for n, s in zip(common, sr) if s > 0.9)

    print(f"\n{'='*80}")
    print(f"  COSINE vs 1/RANK — Unstable Node Comparison (Emb_full, Field 15)")
    print(f"{'='*80}")
    print(f"  Common nodes: {len(common):,}")
    print(f"  Spearman ρ: {rho_s:.3f}")
    print(f"  Pearson r:  {rho_p:.3f}")
    print()
    print(f"  Unstable (<0.5):")
    print(f"    Cosine: {len(unstable_cos):,} ({len(unstable_cos)/len(common)*100:.1f}%)")
    print(f"    1/rank: {len(unstable_rank):,} ({len(unstable_rank)/len(common)*100:.1f}%)")
    print(f"    Overlap: {len(unstable_cos & unstable_rank):,}")
    print(f"    Jaccard: {len(unstable_cos & unstable_rank) / len(unstable_cos | unstable_rank):.3f}")
    print(f"    Only cosine: {len(unstable_cos - unstable_rank):,}")
    print(f"    Only 1/rank: {len(unstable_rank - unstable_cos):,}")
    print()
    print(f"  Stable (>0.9):")
    print(f"    Cosine: {len(stable_cos):,} ({len(stable_cos)/len(common)*100:.1f}%)")
    print(f"    1/rank: {len(stable_rank):,} ({len(stable_rank)/len(common)*100:.1f}%)")
    print(f"    Overlap: {len(stable_cos & stable_rank):,}")
    print(f"    Jaccard: {len(stable_cos & stable_rank) / len(stable_cos | stable_rank):.3f}")
    print()

    # Transition matrix
    cos_cat = np.where(sc > 0.9, "stable", np.where(sc < 0.5, "unstable", "mid"))
    rank_cat = np.where(sr > 0.9, "stable", np.where(sr < 0.5, "unstable", "mid"))
    print(f"  Transition matrix (cosine → 1/rank):")
    print(f"  {'':>15} {'rank_stable':>12} {'rank_mid':>12} {'rank_unstable':>12}")
    for cat in ["stable", "mid", "unstable"]:
        mask = cos_cat == cat
        r_s = np.sum((rank_cat == "stable") & mask)
        r_m = np.sum((rank_cat == "mid") & mask)
        r_u = np.sum((rank_cat == "unstable") & mask)
        total = mask.sum()
        print(f"  cos_{cat:>9} {r_s:>12,} {r_m:>12,} {r_u:>12,}  (total={total:,})")


if __name__ == "__main__":
    main()
