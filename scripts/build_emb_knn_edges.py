#!/usr/bin/env python3
"""Build k-NN edge lists from SPECTER2 embeddings.

Reads h5 embeddings + mapping parquet, computes cosine similarity k-NN,
outputs edge list in the same format as linktype_edges_gcc (uid1, uid2, rel_sum2).

Usage:
    .venv/bin/python scripts/build_emb_knn_edges.py --field 15 --k 30
    .venv/bin/python scripts/build_emb_knn_edges.py --field 12 --k 30 --batch-size 5000
    .venv/bin/python scripts/build_emb_knn_edges.py --field 18 --k 30
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import polars as pl
from scipy import sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_emb_knn")

EMB_DIR = Path.home() / "Desktop/HDD/local_map_analysis_data/processed/outputs/gpu_embeddings/oa26_perfield_specter2_20260331"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
TEXT_DIR = Path(__file__).resolve().parent.parent / "data" / "openalex_metadata"

def _resolve_field_dir(field_id: int) -> Path:
    matches = sorted(EMB_DIR.glob(f"field_{field_id}_*"))
    if not matches:
        raise FileNotFoundError(f"No embedding directory found for field {field_id} under {EMB_DIR}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(f"Multiple embedding directories found for field {field_id}: {names}")
    return matches[0]


def load_embeddings(field_id: int) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and work_id mapping."""
    field_dir = _resolve_field_dir(field_id)

    mapping = pl.read_parquet(field_dir / "mapping.parquet")
    work_ids = mapping["work_id"].to_list()

    with h5py.File(field_dir / "embeddings.h5", "r") as f:
        emb = f["embeddings"][:]  # (N, 768) float32

    if emb.shape[0] != len(work_ids):
        raise ValueError(
            f"Embedding/mapping length mismatch for field {field_id}: "
            f"{emb.shape[0]} embedding rows vs {len(work_ids)} mapping rows"
        )

    log.info("Loaded %d embeddings (dim=%d) for field %d", emb.shape[0], emb.shape[1], field_id)
    return emb, work_ids


def _default_works_text_path(field_id: int) -> Path:
    return TEXT_DIR / f"field_{field_id}" / "works_text.parquet"


def filter_embeddings_by_text_quality(
    emb: np.ndarray,
    work_ids: list[str],
    *,
    field_id: int,
    works_text_path: Path | None = None,
    min_text_len: int = 0,
    min_title_len: int = 0,
    min_abstract_len: int = 0,
    require_abstract: bool = False,
    min_metadata_match: float = 0.95,
    drop_unmatched_metadata: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Filter embeddings using title/abstract quality metadata before k-NN.

    This is a post-processing filter over an existing embedding artifact. Because
    the embedding source and local metadata can come from different OpenAlex
    snapshots, we verify the metadata join rate and fail loudly when it is too low.
    """
    works_text_path = works_text_path or _default_works_text_path(field_id)
    if not works_text_path.exists():
        raise FileNotFoundError(f"works_text parquet not found: {works_text_path}")

    meta = pl.read_parquet(works_text_path, columns=["work_id", "title", "abstract"]).with_columns([
        pl.col("title").fill_null("").cast(pl.String),
        pl.col("abstract").fill_null("").cast(pl.String),
    ]).with_columns([
        pl.col("title").str.len_chars().alias("title_len"),
        pl.col("abstract").str.len_chars().alias("abstract_len"),
    ]).with_columns([
        (pl.col("title_len") + pl.col("abstract_len")).alias("text_len"),
        (
            (pl.col("title_len") >= min_title_len)
            & (pl.col("abstract_len") >= min_abstract_len)
            & (pl.col("title_len") + pl.col("abstract_len") >= min_text_len)
            & ((pl.col("abstract_len") > 0) if require_abstract else pl.lit(True))
        ).alias("keep"),
    ]).select(["work_id", "keep"])

    mapping_df = pl.DataFrame({
        "idx": np.arange(len(work_ids), dtype=np.int64),
        "work_id": work_ids,
    })
    joined = mapping_df.join(meta, on="work_id", how="left")
    matched = joined["keep"].is_not_null().sum()
    match_rate = matched / len(work_ids) if work_ids else 1.0
    if match_rate < min_metadata_match:
        raise ValueError(
            "Metadata join rate too low for text-quality filtering: "
            f"{matched}/{len(work_ids)} = {match_rate:.3f}. "
            "This usually means the embedding artifact and works_text.parquet came "
            "from different snapshots. Pass the matching works_text parquet or "
            "lower --min-metadata-match only if you have verified the source."
        )

    keep_mask = joined["keep"].fill_null(False if drop_unmatched_metadata else True)
    kept = int(keep_mask.sum())
    indices = joined.filter(keep_mask)["idx"].to_list()
    emb = emb[indices]
    work_ids = [work_ids[i] for i in indices]
    stats = {
        "input_nodes": len(joined),
        "matched_nodes": matched,
        "match_rate": match_rate,
        "kept_nodes": kept,
        "dropped_nodes": len(joined) - kept,
        "drop_unmatched_metadata": drop_unmatched_metadata,
        "works_text_path": str(works_text_path),
    }
    log.info(
        "Applied text-quality filter: kept %d / %d nodes (metadata match %.3f) using %s",
        kept,
        len(joined),
        match_rate,
        works_text_path,
    )
    return emb, work_ids, stats


def validate_knn_inputs(n_nodes: int, k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n_nodes <= k:
        raise ValueError(
            f"k={k} requires at least {k + 1} nodes after filtering, got {n_nodes}"
        )


def _filtered_stem(
    k: int,
    *,
    min_text_len: int,
    min_title_len: int,
    min_abstract_len: int,
    require_abstract: bool,
    drop_unmatched_metadata: bool,
) -> str:
    parts = [f"emb_full_knn{k}", "textfilt"]
    if min_text_len:
        parts.append(f"txt{min_text_len}")
    if min_title_len:
        parts.append(f"title{min_title_len}")
    if min_abstract_len:
        parts.append(f"abs{min_abstract_len}")
    if require_abstract:
        parts.append("reqabs")
    if drop_unmatched_metadata:
        parts.append("dropunmatched")
    return "_".join(parts)


def write_filter_manifest(
    manifest_path: Path,
    *,
    field_id: int,
    k: int,
    filter_stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    payload = {
        "field_id": field_id,
        "k": k,
        "filter": {
            "min_text_len": args.min_text_len,
            "min_title_len": args.min_title_len,
            "min_abstract_len": args.min_abstract_len,
            "require_abstract": args.require_abstract,
            "min_metadata_match": args.min_metadata_match,
            "drop_unmatched_metadata": args.drop_unmatched_metadata,
            "works_text_path": str(args.works_text_path) if args.works_text_path else None,
        },
        "stats": filter_stats,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_knn_cosine(
    emb: np.ndarray,
    k: int = 30,
    batch_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build k-NN graph using cosine similarity (batch matmul).

    Returns (src, dst, similarity) arrays — directed (each node's top-k).
    """
    n, d = emb.shape
    log.info("Building k-NN (k=%d) for %d nodes, batch_size=%d", k, n, batch_size)

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb / norms

    all_src = []
    all_dst = []
    all_sim = []

    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = emb_norm[start:end]  # (batch, d)

        # Cosine similarity: batch @ all^T → (batch, n)
        sims = batch @ emb_norm.T  # (batch, n)

        # Zero out self-similarity
        for i in range(end - start):
            sims[i, start + i] = -1.0

        # Top-k per row
        # Use argpartition for efficiency
        topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]  # (batch, k)

        for i in range(end - start):
            global_i = start + i
            neighbors = topk_idx[i]
            neighbor_sims = sims[i, neighbors]

            # Sort by similarity (descending)
            sort_idx = np.argsort(-neighbor_sims)
            neighbors = neighbors[sort_idx]
            neighbor_sims = neighbor_sims[sort_idx]

            all_src.extend([global_i] * k)
            all_dst.extend(neighbors.tolist())
            all_sim.extend(neighbor_sims.tolist())

        if (start // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            pct = end / n * 100
            log.info("  %d/%d (%.1f%%) in %.1fs", end, n, pct, elapsed)

    return np.array(all_src), np.array(all_dst), np.array(all_sim, dtype=np.float32)


def build_knn_faiss(
    emb: np.ndarray,
    k: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build k-NN graph using faiss (much faster for large N).

    Returns (src, dst, similarity) arrays.
    """
    import faiss

    n, d = emb.shape
    log.info("Building k-NN via faiss (k=%d) for %d nodes (dim=%d)", k, n, d)

    # L2 normalize for cosine similarity (faiss IP = cosine on normalized vecs)
    norms = np.linalg.norm(emb, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1.0
    emb_norm = (emb / norms).astype(np.float32)

    # Build index — use IVF for large datasets, Flat for small
    if n > 200_000:
        nlist = min(int(np.sqrt(n)) * 2, 4096)
        nprobe = min(nlist // 16, 64)
        log.info("  Using IVF index: nlist=%d, nprobe=%d", nlist, nprobe)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = nprobe
        log.info("  Training IVF index...")
        t_train = time.time()
        index.train(emb_norm)
        log.info("  Training done in %.1fs", time.time() - t_train)
        index.add(emb_norm)
    else:
        log.info("  Using Flat index (n=%d)", n)
        index = faiss.IndexFlatIP(d)
        index.add(emb_norm)

    # Search k+1 (includes self)
    t0 = time.time()
    similarities, indices = index.search(emb_norm, k + 1)
    log.info("  faiss search done in %.1fs", time.time() - t0)

    # Remove self-matches (vectorized)
    # For each row, find where index == row_id and remove it
    row_ids = np.arange(n).reshape(-1, 1)  # (n, 1)
    self_mask = (indices == row_ids)  # (n, k+1) boolean

    # For rows where self is found, shift columns left
    # Simpler: just mask out self and take first k
    # Set self-match similarity to -inf so it sorts last
    similarities[self_mask] = -np.inf
    indices[self_mask] = -1

    # Sort each row by descending similarity
    sort_idx = np.argsort(-similarities, axis=1)[:, :k]  # (n, k)
    row_idx = np.arange(n).reshape(-1, 1)

    dst_arr = indices[row_idx, sort_idx]  # (n, k)
    sim_arr = similarities[row_idx, sort_idx]  # (n, k)

    # Flatten to 1D arrays
    src_arr = np.repeat(np.arange(n), k)
    dst_flat = dst_arr.ravel()
    sim_flat = sim_arr.ravel().astype(np.float32)

    # Remove any remaining invalid entries (dst == -1)
    valid = dst_flat >= 0
    src_arr = src_arr[valid]
    dst_flat = dst_flat[valid]
    sim_flat = sim_flat[valid]

    log.info("  Post-processing done: %d directed edges", len(src_arr))
    return src_arr, dst_flat, sim_flat


def build_knn_torch(
    emb: np.ndarray,
    k: int = 30,
    batch_size: int = 512,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build k-NN using torch matmul, typically on CUDA."""
    import torch

    n, d = emb.shape
    log.info(
        "Building k-NN via torch (k=%d) for %d nodes (dim=%d, device=%s, batch_size=%d)",
        k,
        n,
        d,
        device,
        batch_size,
    )

    emb_t = torch.from_numpy(emb).to(device=device, dtype=torch.float32)
    emb_t = emb_t / torch.linalg.vector_norm(emb_t, dim=1, keepdim=True).clamp_min(1e-12)

    all_src = []
    all_dst = []
    all_sim = []

    t0 = time.time()
    total_batches = (n + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, n, batch_size), start=1):
        end = min(start + batch_size, n)
        batch = emb_t[start:end]
        sims = batch @ emb_t.T

        row_idx = torch.arange(end - start, device=device)
        col_idx = torch.arange(start, end, device=device)
        sims[row_idx, col_idx] = float("-inf")

        vals, inds = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
        all_src.append(np.repeat(np.arange(start, end), k))
        all_dst.append(inds.cpu().numpy().reshape(-1))
        all_sim.append(vals.cpu().numpy().reshape(-1).astype(np.float32))

        if batch_idx == 1 or batch_idx % 10 == 0 or end == n:
            elapsed = time.time() - t0
            pct = end / n * 100
            log.info("  batch %d/%d, %d/%d (%.1f%%) in %.1fs", batch_idx, total_batches, end, n, pct, elapsed)

    src_arr = np.concatenate(all_src)
    dst_flat = np.concatenate(all_dst)
    sim_flat = np.concatenate(all_sim)
    log.info("  torch search done: %d directed edges", len(src_arr))
    return src_arr, dst_flat, sim_flat


def symmetrize_and_save(
    src: np.ndarray,
    dst: np.ndarray,
    sim: np.ndarray,
    work_ids: list[str],
    out_path: Path,
):
    """Symmetrize directed k-NN and save as edge list."""
    n = len(work_ids)

    # Build sparse matrix (directed)
    M = sparse.csr_matrix((sim, (src, dst)), shape=(n, n))
    # Symmetrize: max of both directions
    M_sym = M.maximum(M.T)

    # Extract upper triangle
    upper = sparse.triu(M_sym, k=1).tocoo()
    mask = upper.data > 0

    log.info("Symmetrized: %d edges (directed %d → undirected %d)",
             len(src), len(src), mask.sum())

    df = pl.DataFrame({
        "uid1": [work_ids[i] for i in upper.row[mask]],
        "uid2": [work_ids[i] for i in upper.col[mask]],
        "rel_sum2": upper.data[mask].astype(np.float32),
    })

    df.write_parquet(out_path)
    log.info("Saved %d edges to %s", df.height, out_path)
    return df


def main():
    parser = argparse.ArgumentParser(description="Build k-NN edges from SPECTER2 embeddings")
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--filter-text", action="store_true",
                        help="Filter embeddings by title/abstract quality before k-NN")
    parser.add_argument("--works-text-path", type=Path, default=None,
                        help="Path to works_text.parquet aligned with the embedding artifact")
    parser.add_argument("--min-text-len", type=int, default=0,
                        help="Minimum combined title+abstract length to keep")
    parser.add_argument("--min-title-len", type=int, default=0,
                        help="Minimum title length to keep")
    parser.add_argument("--min-abstract-len", type=int, default=0,
                        help="Minimum abstract length to keep")
    parser.add_argument("--require-abstract", action="store_true",
                        help="Require a non-empty abstract to keep a node")
    parser.add_argument("--min-metadata-match", type=float, default=0.95,
                        help="Fail if too few embedding work_ids match works_text metadata")
    parser.add_argument("--drop-unmatched-metadata", action="store_true",
                        help="Drop work_ids that do not match works_text metadata after join")
    args = parser.parse_args()

    field_id = args.field
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    emb, work_ids = load_embeddings(field_id)

    # Filter to GCC nodes if node_mapping exists
    mapping_path = out_dir / "node_mapping.parquet"
    if mapping_path.exists():
        gcc_ids = set(pl.read_parquet(mapping_path)["work_id"].to_list())
        # Filter
        mask = [wid in gcc_ids for wid in work_ids]
        indices = [i for i, m in enumerate(mask) if m]
        emb = emb[indices]
        work_ids = [work_ids[i] for i in indices]
        log.info("Filtered to GCC nodes: %d / %d", len(work_ids), len(mask))

    filter_active = (
        args.filter_text
        or args.min_text_len > 0
        or args.min_title_len > 0
        or args.min_abstract_len > 0
        or args.require_abstract
    )
    if filter_active:
        emb, work_ids, filter_stats = filter_embeddings_by_text_quality(
            emb,
            work_ids,
            field_id=field_id,
            works_text_path=args.works_text_path,
            min_text_len=args.min_text_len,
            min_title_len=args.min_title_len,
            min_abstract_len=args.min_abstract_len,
            require_abstract=args.require_abstract,
            min_metadata_match=args.min_metadata_match,
            drop_unmatched_metadata=args.drop_unmatched_metadata,
        )
    else:
        filter_stats = None

    validate_knn_inputs(len(work_ids), args.k)

    # Build k-NN
    t0 = time.time()
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 512 if args.device in {"auto", "cuda"} else 2000

    if args.device in {"auto", "cuda"}:
        try:
            import torch

            if torch.cuda.is_available():
                src, dst, sim = build_knn_torch(emb, k=args.k, batch_size=batch_size, device="cuda")
            elif args.device == "cuda":
                raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is false")
            else:
                raise ImportError
        except ImportError:
            try:
                import faiss  # noqa: F401

                log.info("torch CUDA backend unavailable, falling back to faiss CPU")
                src, dst, sim = build_knn_faiss(emb, k=args.k)
            except ImportError:
                log.info("faiss not available, falling back to batch matmul")
                src, dst, sim = build_knn_cosine(emb, k=args.k, batch_size=batch_size)
    else:
        try:
            import faiss  # noqa: F401

            src, dst, sim = build_knn_faiss(emb, k=args.k)
        except ImportError:
            log.info("faiss not available, falling back to batch matmul")
            src, dst, sim = build_knn_cosine(emb, k=args.k, batch_size=batch_size)
    log.info("k-NN computed in %.1fs", time.time() - t0)

    # Save
    out_stem = (
        _filtered_stem(
            args.k,
            min_text_len=args.min_text_len,
            min_title_len=args.min_title_len,
            min_abstract_len=args.min_abstract_len,
            require_abstract=args.require_abstract,
            drop_unmatched_metadata=args.drop_unmatched_metadata,
        )
        if filter_active
        else f"emb_full_knn{args.k}"
    )
    out_path = out_dir / f"{out_stem}.parquet"
    symmetrize_and_save(src, dst, sim, work_ids, out_path)
    if filter_active and filter_stats is not None:
        write_filter_manifest(
            out_dir / f"{out_stem}.metadata.json",
            field_id=field_id,
            k=args.k,
            filter_stats=filter_stats,
            args=args,
        )


if __name__ == "__main__":
    main()
