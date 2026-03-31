#!/usr/bin/env python3
"""Evaluate clustering quality using text-based metrics (BM25 + embeddings).

Implements:
  1. BM25 accuracy (Waltman et al. 2020 reproduction) — vectorized
  2. Embedding-based accuracy (multiple models)
  3. Cluster-level unweighted accuracy (improved metric M3)
  4. Within–between separation metric

Primary model: SPECTER2 + [PRX] adapter + center(α=0.66) from Nanocluster project.
Robustness check: all-mpnet-base-v2 (general-purpose).

Usage:
    python scripts/eval_text_quality.py --field 34
    python scripts/eval_text_quality.py --field 34 --skip-prx
    python scripts/eval_text_quality.py --field 34 --skip-bm25 --skip-st
    python scripts/eval_text_quality.py --field 34 --models all-mpnet-base-v2
"""
from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_text")

DATA_ROOT = Path.home() / "Desktop/HDD/local_map_analysis_data/processed/outputs/data"
NODE_DIR = DATA_ROOT / "oa26_gcc_only_k30"
EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges"
TEXT_DIR = Path(__file__).resolve().parent.parent / "data" / "openalex_metadata"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results"

# BM25 parameters (Waltman et al. 2020)
BM25_K1 = 2.0
BM25_B = 0.75

# [PRX] adapter path (Nanocluster project publish-ready bundle)
PRX_ADAPTER_DIR = (
    Path.home() / "Desktop/Workspace/1.1.4.KISTI_NanoClustering"
    / "outputs/hf_upload"
    / "stage2_policy_p2s_r60_40_prx_cont_center066_20260311_publish_ready"
)
PRX_CTRL_TOKEN = "[PRX]"

# Sentence-transformer models for robustness check
DEFAULT_ST_MODELS = [
    "all-mpnet-base-v2",
]

N_LEIDEN_RUNS = 5
GAMMA_VALUES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_texts(field_id: int) -> pl.DataFrame:
    path = TEXT_DIR / f"field_{field_id}" / "works_text.parquet"
    df = pl.read_parquet(path, columns=["work_id", "title", "abstract"])
    df = df.with_columns(
        pl.when(pl.col("abstract").is_not_null() & (pl.col("abstract") != ""))
        .then(pl.col("title").fill_null("") + " " + pl.col("abstract").fill_null(""))
        .otherwise(pl.col("title").fill_null(""))
        .alias("text")
    )
    return df


def load_graph_and_cluster(field_id: int, link_type: str, gamma: float, seed: int = 0):
    import igraph as ig
    import leidenalg

    path = EDGE_DIR / f"field_{field_id}" / f"{link_type}.parquet"
    df = pl.read_parquet(path)

    all_ids = pl.concat([df["src"], df["dst"]]).unique().sort().to_list()
    id2idx = {wid: i for i, wid in enumerate(all_ids)}
    src = [id2idx[s] for s in df["src"].to_list()]
    dst = [id2idx[d] for d in df["dst"].to_list()]
    weights = df["weight"].to_list()

    g = ig.Graph(n=len(all_ids), edges=list(zip(src, dst)), directed=False)
    g.es["weight"] = weights
    g = g.simplify(combine_edges="max")

    # Waltman ExtDC: load node_sizes (focal=1, non-focal=0)
    node_sizes = None
    focal_set = None
    nodes_path = EDGE_DIR / f"field_{field_id}" / f"{link_type}_nodes.parquet"
    if nodes_path.exists():
        ndf = pl.read_parquet(nodes_path)
        focal_map = dict(zip(ndf["work_id"].to_list(), ndf["is_focal"].to_list()))
        node_sizes = [focal_map.get(wid, 1) for wid in all_ids]
        focal_set = {wid for wid, f in focal_map.items() if f == 1}
        log.info("  Loaded node_sizes: %d focal, %d non-focal",
                 sum(node_sizes), len(node_sizes) - sum(node_sizes))

    gcc_idx = g.connected_components().giant().vs.indices
    g_gcc = g.subgraph(gcc_idx)
    all_work_ids = [all_ids[gcc_idx[i]] for i in range(g_gcc.vcount())]

    gcc_node_sizes = None
    if node_sizes is not None:
        gcc_node_sizes = [node_sizes[gcc_idx[i]] for i in range(g_gcc.vcount())]

    part = leidenalg.find_partition(
        g_gcc, leidenalg.CPMVertexPartition,
        resolution_parameter=gamma, weights="weight", seed=seed,
        **({"node_sizes": gcc_node_sizes} if gcc_node_sizes else {}),
    )

    # For Waltman ExtDC: return only focal nodes and their membership
    if focal_set is not None:
        focal_wids = []
        focal_mem = []
        for i, wid in enumerate(all_work_ids):
            if wid in focal_set:
                focal_wids.append(wid)
                focal_mem.append(part.membership[i])
        return focal_wids, focal_mem, g_gcc

    return all_work_ids, part.membership, g_gcc


# ──────────────────────────────────────────────
# BM25 — fully vectorized
# ──────────────────────────────────────────────

class BM25Index:
    """Precomputed BM25 index for fast within-cluster scoring."""

    def __init__(self, texts: list[str], max_features: int = 50000):
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )
        self.tf = vectorizer.fit_transform(texts).tocsr()  # (N, V)
        N, V = self.tf.shape

        doc_lens = np.array(self.tf.sum(axis=1)).flatten()
        avg_dl = doc_lens.mean()

        # IDF
        df_vec = np.array((self.tf > 0).sum(axis=0)).flatten()
        idf = np.log((N - df_vec + 0.5) / (df_vec + 0.5))
        idf = np.maximum(idf, 0)

        # Precompute BM25 tf component: tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
        # This is per-document per-term
        dl_factor = BM25_K1 * (1 - BM25_B + BM25_B * doc_lens / avg_dl)  # (N,)

        # Build BM25-weighted tf matrix (sparse)
        tf_coo = self.tf.tocoo()
        bm25_data = (
            tf_coo.data * (BM25_K1 + 1)
            / (tf_coo.data + dl_factor[tf_coo.row])
        )
        self.bm25_tf = sp.csr_matrix(
            (bm25_data, (tf_coo.row, tf_coo.col)), shape=(N, V)
        )

        # IDF-weighted BM25 tf: multiply each column by IDF
        # bm25_weighted[i,l] = bm25_tf[i,l] * idf[l]
        self.bm25_weighted = self.bm25_tf.multiply(idf[np.newaxis, :]).tocsr()

        # Binary presence matrix for queries
        self.presence = (self.tf > 0).astype(np.float32).tocsr()

        log.info("  BM25 index: %d docs, %d terms", N, V)

    def score_cluster(self, indices: np.ndarray, max_k: int = 500) -> float:
        """Compute mean symmetrized BM25 for pairs in a cluster.

        For clusters > max_k, subsample to keep computation tractable.
        """
        k = len(indices)
        if k < 2:
            return 0.0

        # Subsample large clusters
        if k > max_k:
            rng = np.random.RandomState(indices[0])
            sample = rng.choice(k, max_k, replace=False)
            indices = indices[sample]
            k = max_k

        pres_sub = self.presence[indices]       # (k, V) sparse
        bm25_sub = self.bm25_weighted[indices]  # (k, V) sparse

        score_mat = (pres_sub @ bm25_sub.T).toarray()  # (k, k) dense
        sym = (score_mat + score_mat.T) / 2
        pair_sum = (sym.sum() - np.trace(sym)) / 2
        n_pairs = k * (k - 1) // 2
        return float(pair_sum / n_pairs)

    def within_cluster_accuracy(self, membership: list[int]) -> dict:
        """Waltman-style BM25 accuracy + unweighted cluster-level."""
        N = len(membership)
        clusters = {}
        for i, c in enumerate(membership):
            clusters.setdefault(c, []).append(i)

        total_pair_score = 0.0
        cluster_means = []

        for cid, members in clusters.items():
            if len(members) < 2:
                continue
            idx = np.array(members)
            mean_score = self.score_cluster(idx)
            n_pairs = len(members) * (len(members) - 1) // 2
            total_pair_score += mean_score * n_pairs
            cluster_means.append(mean_score)

        accuracy = total_pair_score / N if N > 0 else 0
        granularity = N / sum(len(m) ** 2 for m in clusters.values()) if clusters else 0
        unweighted = float(np.mean(cluster_means)) if cluster_means else 0

        return {
            "bm25_accuracy": float(accuracy),
            "bm25_granularity": float(granularity),
            "bm25_unweighted": unweighted,
        }


# ──────────────────────────────────────────────
# [PRX] adapter embedding (primary model)
# ──────────────────────────────────────────────

def compute_prx_embeddings(
    texts: list[str],
    adapter_dir: Path = PRX_ADAPTER_DIR,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode texts using SPECTER2 + [PRX] adapter + centering on GPU."""
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("  Loading SPECTER2 + [PRX] adapter on %s...", device)

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base", use_fast=True)
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    tokenizer.add_special_tokens({"additional_special_tokens": [PRX_CTRL_TOKEN]})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model.resize_token_embeddings(len(tokenizer))
    model.load_adapter(str(adapter_dir), load_as=PRX_CTRL_TOKEN, set_active=True)
    model.to(device).eval()

    # Load centering postprocess
    pp_path = adapter_dir / "postprocess" / "stage2_center_alpha066_20260307.pt"
    artifact = torch.load(pp_path, map_location=device)
    mean_vec = artifact["mean"].to(device)
    strength = float(artifact["strength"])
    log.info("  Adapter loaded, center α=%.2f", strength)

    # Encode
    log.info("  Encoding %d texts...", len(texts))
    all_embs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch_texts = [f"{PRX_CTRL_TOKEN} {t[:500]}" for t in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=512,
            return_tensors="pt", return_token_type_ids=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, return_dict=True)
            emb = torch.nn.functional.normalize(out.last_hidden_state[:, 1], p=2, dim=-1)
            emb = torch.nn.functional.normalize(emb - (strength * mean_vec), p=2, dim=-1)
            all_embs.append(emb.cpu().numpy())
        if (i // batch_size) % 20 == 0:
            log.info("    batch %d/%d", i // batch_size, n_batches)

    embeddings = np.vstack(all_embs)
    log.info("  [PRX] done: %s", embeddings.shape)

    # Free GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return embeddings


# ──────────────────────────────────────────────
# Sentence-transformer embeddings (robustness)
# ──────────────────────────────────────────────

def compute_st_embeddings(texts: list[str], model_name: str, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    log.info("  Loading model: %s", model_name)
    model = SentenceTransformer(model_name)
    log.info("  Encoding %d texts...", len(texts))
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


# ──────────────────────────────────────────────
# Embedding cluster metrics
# ──────────────────────────────────────────────

def embedding_cluster_metrics(
    embeddings: np.ndarray,
    membership: list[int],
    max_k: int = 1000,
    n_between_samples: int = 5000,
) -> dict:
    """Compute embedding-based accuracy, unweighted accuracy, separation."""
    N = len(membership)
    clusters = {}
    for i, c in enumerate(membership):
        clusters.setdefault(c, []).append(i)

    # ── Within-cluster accuracy (subsample large clusters) ──
    total_pair_sim = 0.0
    cluster_means = []

    for cid, members in clusters.items():
        k = len(members)
        if k < 2:
            continue
        idx = np.array(members)
        if k > max_k:
            rng = np.random.RandomState(cid)
            idx = idx[rng.choice(k, max_k, replace=False)]
        emb_c = embeddings[idx]
        sim_mat = emb_c @ emb_c.T
        ks = len(idx)
        pair_sum = (sim_mat.sum() - np.trace(sim_mat)) / 2
        n_pairs = ks * (ks - 1) // 2
        mean_sim = float(pair_sum / n_pairs)
        actual_pairs = k * (k - 1) // 2
        total_pair_sim += mean_sim * actual_pairs
        cluster_means.append(mean_sim)

    accuracy = float(total_pair_sim / N) if N > 0 else 0
    granularity = N / sum(len(m) ** 2 for m in clusters.values()) if clusters else 0
    unweighted = float(np.mean(cluster_means)) if cluster_means else 0

    # ── Between-cluster similarity (sampled) ──
    big_clusters = {c: m for c, m in clusters.items() if len(m) >= 5}
    if len(big_clusters) >= 2:
        rng = np.random.RandomState(42)
        cids = list(big_clusters.keys())
        between_sims = []
        for _ in range(n_between_samples):
            c1, c2 = rng.choice(len(cids), 2, replace=False)
            i1 = rng.choice(big_clusters[cids[c1]])
            i2 = rng.choice(big_clusters[cids[c2]])
            between_sims.append(float(embeddings[i1] @ embeddings[i2]))
        between = float(np.mean(between_sims))
    else:
        between = float("nan")

    separation = unweighted - between

    return {
        "accuracy": accuracy,
        "granularity": granularity,
        "unweighted": unweighted,
        "between": between,
        "separation": separation,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering quality with text metrics")
    parser.add_argument("--field", type=int, default=34)
    parser.add_argument("--link-types", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=DEFAULT_ST_MODELS,
                        help="Sentence-transformer models for robustness check")
    parser.add_argument("--gammas", nargs="*", type=float, default=GAMMA_VALUES)
    parser.add_argument("--n-runs", type=int, default=N_LEIDEN_RUNS)
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--skip-prx", action="store_true", help="Skip [PRX] adapter")
    parser.add_argument("--skip-st", action="store_true", help="Skip sentence-transformer models")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    # ── Load texts ──
    log.info("Loading texts for field %d...", args.field)
    text_df = load_texts(args.field)
    texts = text_df["text"].to_list()
    work_ids_text = text_df["work_id"].to_list()
    wid_to_idx = {wid: i for i, wid in enumerate(work_ids_text)}
    log.info("  %d texts loaded", len(texts))

    # ── Precompute BM25 index ──
    bm25 = None
    if not args.skip_bm25:
        log.info("Building BM25 index...")
        t0 = time.time()
        bm25 = BM25Index(texts)
        log.info("  Done in %.1fs", time.time() - t0)

    # ── Precompute embeddings ──
    emb_map = {}  # name → (embeddings, short_name)

    # Primary: [PRX] adapter
    if not args.skip_prx:
        log.info("Computing [PRX] adapter embeddings (primary)...")
        t0 = time.time()
        try:
            emb_prx = compute_prx_embeddings(texts, batch_size=args.batch_size)
            emb_map["prx_center066"] = emb_prx
            log.info("  [PRX]: dim=%d, %.1fs", emb_prx.shape[1], time.time() - t0)
        except Exception as e:
            log.error("  Failed [PRX]: %s", e)

    # Robustness: sentence-transformer models
    if not args.skip_st:
        for model_name in args.models:
            t0 = time.time()
            try:
                emb = compute_st_embeddings(texts, model_name, args.batch_size)
                short = model_name.split("/")[-1]
                emb_map[short] = emb
                log.info("  %s: dim=%d, %.1fs", short, emb.shape[1], time.time() - t0)
            except Exception as e:
                log.error("  Failed %s: %s", model_name, e)

    # ── Discover link types ──
    edge_dir = EDGE_DIR / f"field_{args.field}"
    if not edge_dir.exists():
        log.error("No edge data for field %d", args.field)
        return

    if args.link_types:
        link_types = args.link_types
    else:
        link_types = sorted(
            p.stem for p in edge_dir.glob("*.parquet")
            if p.stem != "node_mapping"
            and not p.stem.endswith("_nodes")
        )

    log.info("Evaluating %d link types × %d γ values × %d runs",
             len(link_types), len(args.gammas), args.n_runs)

    # ── Evaluate ──
    all_results = []
    total_combos = len(link_types) * len(args.gammas)
    combo_idx = 0

    for lt in link_types:
        for gamma in args.gammas:
            combo_idx += 1
            t_combo = time.time()
            run_results = []

            for seed in range(args.n_runs):
                try:
                    work_ids, membership, g = load_graph_and_cluster(
                        args.field, lt, gamma, seed,
                    )
                except Exception as e:
                    log.warning("  %s γ=%.5f seed=%d: %s", lt, gamma, seed, e)
                    continue

                # Map to text indices
                text_idx = []
                valid_graph = []
                for i, wid in enumerate(work_ids):
                    if wid in wid_to_idx:
                        text_idx.append(wid_to_idx[wid])
                        valid_graph.append(i)
                text_idx = np.array(text_idx)
                valid_mem = [membership[i] for i in valid_graph]

                row = {
                    "link_type": lt,
                    "gamma": gamma,
                    "seed": seed,
                    "n_nodes": len(valid_graph),
                    "n_clusters": len(set(valid_mem)),
                }

                # BM25
                if bm25 is not None:
                    clusters_txt = {}
                    for gi, ti in zip(valid_graph, text_idx):
                        c = membership[gi]
                        clusters_txt.setdefault(c, []).append(ti)

                    total_ps = 0.0
                    c_means = []
                    N_valid = len(valid_graph)
                    for cid, members_ti in clusters_txt.items():
                        if len(members_ti) < 2:
                            continue
                        ms = bm25.score_cluster(np.array(members_ti))
                        n_pairs = len(members_ti) * (len(members_ti) - 1) // 2
                        total_ps += ms * n_pairs
                        c_means.append(ms)

                    row["bm25_accuracy"] = float(total_ps / N_valid) if N_valid > 0 else 0
                    row["bm25_granularity"] = (
                        N_valid / sum(len(m) ** 2 for m in clusters_txt.values())
                        if clusters_txt else 0
                    )
                    row["bm25_unweighted"] = float(np.mean(c_means)) if c_means else 0

                # Embedding models
                for model_key, all_emb in emb_map.items():
                    emb_sub = all_emb[text_idx]
                    eres = embedding_cluster_metrics(emb_sub, valid_mem)
                    for mk, mv in eres.items():
                        row[f"{model_key}_{mk}"] = mv

                run_results.append(row)

            # Average across seeds
            if run_results:
                avg = {"link_type": lt, "gamma": gamma}
                skip_keys = {"link_type", "gamma", "seed"}
                numeric_keys = [k for k in run_results[0] if k not in skip_keys]
                for k in numeric_keys:
                    vals = [r[k] for r in run_results if k in r]
                    avg[k] = float(np.mean(vals))
                    avg[f"{k}_std"] = float(np.std(vals))
                all_results.append(avg)

                log.info("  [%d/%d] %s γ=%.5f: %d nodes, %d clusters (%.1fs)",
                         combo_idx, total_combos, lt, gamma,
                         int(avg["n_nodes"]), int(avg["n_clusters"]),
                         time.time() - t_combo)

    # ── Save ──
    if all_results:
        out_dir = OUT_DIR / f"field_{args.field}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df = pl.DataFrame(all_results)
        out_path = out_dir / "text_quality_results.parquet"

        # Merge with existing results (don't overwrite other link types)
        if out_path.exists():
            existing = pl.read_parquet(out_path)
            new_lts = set(results_df["link_type"].unique().to_list())
            kept = existing.filter(~pl.col("link_type").is_in(new_lts))
            if len(kept) > 0:
                # Align columns before concat
                for col in results_df.columns:
                    if col not in kept.columns:
                        kept = kept.with_columns(pl.lit(None).cast(results_df[col].dtype).alias(col))
                for col in kept.columns:
                    if col not in results_df.columns:
                        results_df = results_df.with_columns(pl.lit(None).cast(kept[col].dtype).alias(col))
                results_df = pl.concat([kept, results_df], how="diagonal_relaxed")
                log.info("  Merged with %d existing rows (%d new)",
                         len(kept), len(results_df) - len(kept))

        results_df.write_parquet(out_path)
        log.info("Saved → %s", out_path)
        print_results(results_df, emb_map.keys())
    else:
        log.warning("No results produced")


def print_results(df: pl.DataFrame, model_keys):
    """Print compact summary table."""
    cols = df.columns

    print(f"\n{'='*140}")
    print(f"TEXT-BASED QUALITY EVALUATION")
    print(f"{'='*140}")

    # Build header
    metric_groups = []
    if "bm25_accuracy" in cols:
        metric_groups.append(("BM25", "bm25"))
    for mk in model_keys:
        if f"{mk}_unweighted" in cols:
            label = mk[:20]
            metric_groups.append((label, mk))

    header = f"{'Link Type':<25} {'γ':>8} {'N':>6} {'#C':>5}"
    for label, _ in metric_groups:
        header += f" │ {label+'_UW':>12} {label+'_Sep':>12}"
    print(header)
    print("-" * len(header))

    for row in df.sort(["link_type", "gamma"]).iter_rows(named=True):
        line = f"{row['link_type']:<25} {row['gamma']:>8.5f} {row['n_nodes']:>6.0f} {row['n_clusters']:>5.0f}"
        for label, prefix in metric_groups:
            uw = row.get(f"{prefix}_unweighted", 0)
            sep = row.get(f"{prefix}_separation", 0)
            if prefix == "bm25":
                line += f" │ {uw:>12.4f} {'n/a':>12}"
            else:
                line += f" │ {uw:>12.4f} {sep:>12.4f}"
        print(line)


if __name__ == "__main__":
    main()
