#!/usr/bin/env python3
"""Build k-NN edge lists from SPECTER2 embeddings using faiss-gpu.

Designed to run on GPU server. Reads H5 embeddings, builds k-NN via
faiss GpuIndexFlatIP, outputs parquet edge lists.

Usage:
    # Full embeddings k-NN for field 12
    python gpu_build_emb_knn.py --field 12 --k 30 --gpu 0

    # Full embeddings k-NN for field 15
    python gpu_build_emb_knn.py --field 15 --k 30 --gpu 0

    # Additional fields resolve automatically from field_{id}_* directories
    python gpu_build_emb_knn.py --field 18 --k 30 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import faiss
import h5py
import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gpu_knn")

# Paths on GPU server's HDD (mounted)
EMB_DIR = Path("/home/master/Desktop/HDD/local_map_analysis_data/processed/outputs/gpu_embeddings/oa26_perfield_specter2_20260331")
# Fallback: check local HDD too
EMB_DIR_ALT = Path.home() / "Desktop/HDD/local_map_analysis_data/processed/outputs/gpu_embeddings/oa26_perfield_specter2_20260331"

OUT_DIR = Path("/tmp/emb_knn_output")
TEXT_DIR = Path(__file__).resolve().parent.parent / "data" / "openalex_metadata"

def _resolve_field_dir(emb_dir: Path, field_id: int) -> Path:
    matches = sorted(emb_dir.glob(f"field_{field_id}_*"))
    if not matches:
        raise FileNotFoundError(f"No embedding directory found for field {field_id} under {emb_dir}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(f"Multiple embedding directories found for field {field_id}: {names}")
    return matches[0]


def find_emb_dir() -> Path:
    """Find the embeddings directory."""
    for d in [EMB_DIR, EMB_DIR_ALT]:
        if d.exists():
            return d
    raise FileNotFoundError(f"Embeddings not found at {EMB_DIR} or {EMB_DIR_ALT}")


def load_embeddings(field_id: int, gcc_ids: set[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and work_id mapping, optionally filter to GCC."""
    emb_dir = find_emb_dir()
    field_dir = _resolve_field_dir(emb_dir, field_id)

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

    # Filter to GCC if provided
    if gcc_ids is not None:
        mask = [wid in gcc_ids for wid in work_ids]
        indices = [i for i, m in enumerate(mask) if m]
        emb = emb[indices]
        work_ids = [work_ids[i] for i in indices]
        log.info("Filtered to GCC: %d / %d nodes", len(work_ids), len(mask))

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


def build_knn_gpu(
    emb: np.ndarray,
    k: int = 30,
    gpu_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build k-NN using faiss GPU (cosine via normalized IP)."""
    n, d = emb.shape
    log.info("Building k-NN via faiss-gpu (k=%d, n=%d, d=%d, gpu=%d)", k, n, d, gpu_id)

    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1.0
    emb_norm = (emb / norms).astype(np.float32)

    # GPU resources
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = gpu_id

    # Build GPU index
    index = faiss.GpuIndexFlatIP(res, d, config)
    index.add(emb_norm)
    log.info("Index built, searching...")

    t0 = time.time()
    similarities, indices = index.search(emb_norm, k + 1)
    log.info("Search done in %.1fs", time.time() - t0)

    # Remove self-matches (vectorized)
    row_ids = np.arange(n).reshape(-1, 1)
    self_mask = (indices == row_ids)
    similarities[self_mask] = -np.inf
    indices[self_mask] = -1

    # Sort and take top-k
    sort_idx = np.argsort(-similarities, axis=1)[:, :k]
    row_idx = np.arange(n).reshape(-1, 1)
    dst_arr = indices[row_idx, sort_idx]
    sim_arr = similarities[row_idx, sort_idx]

    # Flatten
    src_arr = np.repeat(np.arange(n), k)
    dst_flat = dst_arr.ravel()
    sim_flat = sim_arr.ravel().astype(np.float32)

    # Remove invalid
    valid = dst_flat >= 0
    src_arr = src_arr[valid]
    dst_flat = dst_flat[valid]
    sim_flat = sim_flat[valid]

    log.info("Directed edges: %d", len(src_arr))
    return src_arr, dst_flat, sim_flat


def symmetrize_and_save(
    src: np.ndarray,
    dst: np.ndarray,
    sim: np.ndarray,
    work_ids: list[str],
    out_path: Path,
):
    """Symmetrize and save as parquet."""
    from scipy import sparse

    n = len(work_ids)
    M = sparse.csr_matrix((sim, (src, dst)), shape=(n, n))
    M_sym = M.maximum(M.T)

    upper = sparse.triu(M_sym, k=1).tocoo()
    mask = upper.data > 0

    df = pl.DataFrame({
        "uid1": [work_ids[i] for i in upper.row[mask]],
        "uid2": [work_ids[i] for i in upper.col[mask]],
        "rel_sum2": upper.data[mask].astype(np.float32),
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    log.info("Saved %d edges to %s", df.height, out_path)


def main():
    parser = argparse.ArgumentParser(description="Build k-NN edges on GPU")
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gcc-mapping", type=str, default=None,
                        help="Path to node_mapping.parquet for GCC filtering")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--filter-text", action="store_true",
                        help="Filter embeddings by title/abstract quality before k-NN")
    parser.add_argument("--works-text-path", type=Path, default=None,
                        help="Path to works_text.parquet aligned with the embedding artifact")
    parser.add_argument("--min-text-len", type=int, default=0)
    parser.add_argument("--min-title-len", type=int, default=0)
    parser.add_argument("--min-abstract-len", type=int, default=0)
    parser.add_argument("--require-abstract", action="store_true")
    parser.add_argument("--min-metadata-match", type=float, default=0.95)
    parser.add_argument("--drop-unmatched-metadata", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR / f"field_{args.field}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GCC IDs if provided
    gcc_ids = None
    if args.gcc_mapping:
        gcc_ids = set(pl.read_parquet(args.gcc_mapping)["work_id"].to_list())
        log.info("GCC filter: %d nodes", len(gcc_ids))

    # Load and build
    emb, work_ids = load_embeddings(args.field, gcc_ids)
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
            field_id=args.field,
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

    t0 = time.time()
    src, dst, sim = build_knn_gpu(emb, k=args.k, gpu_id=args.gpu)
    log.info("Total k-NN time: %.1fs", time.time() - t0)

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
            field_id=args.field,
            k=args.k,
            filter_stats=filter_stats,
            args=args,
        )

    log.info("Done!")


if __name__ == "__main__":
    main()
