#!/usr/bin/env python3
"""Build k-NN graph from text embeddings for comparison with citation networks.

Creates edge files in the same format as linktype_edges (src, dst, weight)
so they can be evaluated by eval_text_quality.py directly.

Models:
  - PRX: SPECTER2 + [PRX] adapter + center(α=0.66)  [science-specific]
  - MPNet: all-mpnet-base-v2                          [general-purpose]

Usage:
    python scripts/build_embedding_knn_edges.py --field 34
    python scripts/build_embedding_knn_edges.py --field 34 --k 30 --skip-mpnet
    python scripts/build_embedding_knn_edges.py --fields 34 12
"""
from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("emb_knn")

TEXT_DIR = Path(__file__).resolve().parent.parent / "data" / "openalex_metadata"
EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges"

# [PRX] adapter
PRX_ADAPTER_DIR = (
    Path.home() / "Desktop/Workspace/1.1.4.KISTI_NanoClustering"
    / "outputs/hf_upload"
    / "stage2_policy_p2s_r60_40_prx_cont_center066_20260311_publish_ready"
)
PRX_CTRL_TOKEN = "[PRX]"


def load_texts(field_id: int) -> tuple[list[str], list[str]]:
    """Load texts and work_ids for a field."""
    path = TEXT_DIR / f"field_{field_id}" / "works_text.parquet"
    df = pl.read_parquet(path, columns=["work_id", "title", "abstract"])
    df = df.with_columns(
        pl.when(pl.col("abstract").is_not_null() & (pl.col("abstract") != ""))
        .then(pl.col("title").fill_null("") + " " + pl.col("abstract").fill_null(""))
        .otherwise(pl.col("title").fill_null(""))
        .alias("text")
    )
    return df["text"].to_list(), df["work_id"].to_list()


def compute_prx_embeddings(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Encode with SPECTER2 + [PRX] adapter + centering on GPU."""
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
    model.load_adapter(str(PRX_ADAPTER_DIR), load_as=PRX_CTRL_TOKEN, set_active=True)
    model.to(device).eval()

    pp_path = PRX_ADAPTER_DIR / "postprocess" / "stage2_center_alpha066_20260307.pt"
    artifact = torch.load(pp_path, map_location=device)
    mean_vec = artifact["mean"].to(device)
    strength = float(artifact["strength"])

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
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return embeddings


def compute_mpnet_embeddings(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Encode with all-mpnet-base-v2."""
    from sentence_transformers import SentenceTransformer

    log.info("  Loading all-mpnet-base-v2...")
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


def build_knn_edges(
    embeddings: np.ndarray,
    work_ids: list[str],
    k: int = 30,
    batch_size: int = 2048,
) -> pl.DataFrame:
    """Build symmetric k-NN graph from normalized embeddings.

    For each node, find top-k nearest neighbors by cosine similarity.
    Symmetric: if A is in B's top-k OR B is in A's top-k, edge exists.
    Weight = cosine similarity.
    """
    import torch

    N = len(work_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_t = torch.from_numpy(embeddings).to(device)

    log.info("  Building k-NN (k=%d) for %d nodes on %s...", k, N, device)

    edges = set()
    edge_weights = {}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        # (batch, dim) @ (dim, N) -> (batch, N)
        sim = emb_t[start:end] @ emb_t.T  # cosine sim (already normalized)
        # Zero out self-similarity
        for i in range(start, end):
            sim[i - start, i] = -1.0

        # Top-k per row
        topk_vals, topk_idx = torch.topk(sim, k, dim=1)
        topk_vals = topk_vals.cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()

        for bi in range(end - start):
            gi = start + bi
            for j_pos in range(k):
                gj = int(topk_idx[bi, j_pos])
                w = float(topk_vals[bi, j_pos])
                edge_key = (min(gi, gj), max(gi, gj))
                if edge_key not in edge_weights or w > edge_weights[edge_key]:
                    edge_weights[edge_key] = w
                edges.add(edge_key)

        if start % (batch_size * 4) == 0:
            log.info("    processed %d/%d nodes, %d edges so far", end, N, len(edges))

    del emb_t
    torch.cuda.empty_cache()

    # Convert to DataFrame
    src_list, dst_list, w_list = [], [], []
    for (i, j), w in edge_weights.items():
        src_list.append(work_ids[i])
        dst_list.append(work_ids[j])
        w_list.append(w)

    df = pl.DataFrame({
        "src": src_list,
        "dst": dst_list,
        "weight": w_list,
    })
    log.info("  k-NN graph: %d nodes, %d edges", N, len(df))
    return df


def process_field(field_id: int, k: int, batch_size: int,
                  skip_prx: bool, skip_mpnet: bool):
    """Build embedding k-NN edges for one field."""
    log.info("=== Field %d ===", field_id)
    texts, work_ids = load_texts(field_id)
    log.info("  %d texts loaded", len(texts))

    out_dir = EDGE_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_prx:
        log.info("Computing PRX embeddings...")
        t0 = time.time()
        emb_prx = compute_prx_embeddings(texts, batch_size)
        log.info("  PRX embeddings: %s in %.1fs", emb_prx.shape, time.time() - t0)

        log.info("Building PRX k-NN edges...")
        t0 = time.time()
        df_prx = build_knn_edges(emb_prx, work_ids, k=k)
        out_path = out_dir / f"emb_prx_knn{k}.parquet"
        df_prx.write_parquet(out_path)
        log.info("  Saved → %s (%.1fs)", out_path, time.time() - t0)
        del emb_prx, df_prx
        gc.collect()

    if not skip_mpnet:
        log.info("Computing MPNet embeddings...")
        t0 = time.time()
        emb_mpnet = compute_mpnet_embeddings(texts, batch_size)
        log.info("  MPNet embeddings: %s in %.1fs", emb_mpnet.shape, time.time() - t0)

        log.info("Building MPNet k-NN edges...")
        t0 = time.time()
        df_mpnet = build_knn_edges(emb_mpnet, work_ids, k=k)
        out_path = out_dir / f"emb_mpnet_knn{k}.parquet"
        df_mpnet.write_parquet(out_path)
        log.info("  Saved → %s (%.1fs)", out_path, time.time() - t0)
        del emb_mpnet, df_mpnet
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Build embedding k-NN edge files")
    parser.add_argument("--fields", type=int, nargs="+", default=[34, 12])
    parser.add_argument("--k", type=int, default=30, help="Number of nearest neighbors")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-prx", action="store_true")
    parser.add_argument("--skip-mpnet", action="store_true")
    args = parser.parse_args()

    for field_id in args.fields:
        process_field(field_id, args.k, args.batch_size,
                      args.skip_prx, args.skip_mpnet)

    log.info("Done!")


if __name__ == "__main__":
    main()
