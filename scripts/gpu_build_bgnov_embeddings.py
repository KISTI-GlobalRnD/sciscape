#!/usr/bin/env python3
"""Build bg/nov split embeddings on GPU server using SPECTER2+PRX adapter.

Pipeline:
  1. Load docs (title + abstract) for a field
  2. Split abstract into bg/nov sentences (role_fields.py)
  3. Encode bg text and nov text separately with SPECTER2
  4. Build k-NN edges for each (faiss)
  5. Save as parquet

Usage:
    python gpu_build_bgnov_embeddings.py --field 15 --k 30 --gpu 0
    python gpu_build_bgnov_embeddings.py --field 12 --k 30 --gpu 0 --batch-size 256
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import faiss
import h5py
import numpy as np
import polars as pl
from scipy import sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bgnov")

DOCS_DIR = Path("/data/openalex_embeddings/oa20new_20260313/extracted")
DOCS_FILES = {
    12: "field_12_arts_docs_20260313.parquet",
    15: "field_15_chemeng_docs_20260313.parquet",
}

# ── Sentence role classification (inline from role_fields.py) ─────

import re

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

_BG_CUES = [
    "has been", "have been", "has attracted", "have attracted",
    "is known", "are known", "was reported", "were reported",
    "was examined", "were examined", "was studied", "were studied",
    "is widely used", "are widely used", "is important", "is essential",
    "previous studies", "prior work", "traditionally", "conventionally",
    "it is well known", "it is well established",
    "in recent years", "in the past decade", "recently,",
]

_NOV_CUES = [
    "we propose", "we present", "we develop", "we introduce",
    "we report", "we demonstrate", "we show that", "we found",
    "this paper", "this study", "this work", "in this letter",
    "our method", "our approach", "our results",
    "results show", "our analysis show",
    "a novel", "a new", "for the first time",
    "here we", "here,",
]


def _sentence_role(sent: str, idx: int, total: int) -> str:
    """Classify a sentence as 'bg' or 'nov'."""
    s = sent.lower().strip()
    # First sentence is usually background
    if idx == 0:
        return "bg"
    # Last sentence is usually conclusion/novelty
    if idx == total - 1 and total > 2:
        return "nov"
    # Cue word matching
    for cue in _NOV_CUES:
        if cue in s:
            return "nov"
    for cue in _BG_CUES:
        if cue in s:
            return "bg"
    # Positional fallback: first half bg, second half nov
    if idx < total / 2:
        return "bg"
    return "nov"


def split_bg_nov(title: str, abstract: str) -> tuple[str, str]:
    """Split title+abstract into bg and nov text.

    Returns (bg_text, nov_text) where each is a concatenation of
    relevant sentences prefixed with title.
    """
    if not abstract or not isinstance(abstract, str) or len(abstract.strip()) < 20:
        # No abstract: title goes to both
        t = title if title else ""
        return t, t

    sentences = _SENT_SPLIT.split(abstract.strip())
    total = len(sentences)

    bg_sents = []
    nov_sents = []
    for i, sent in enumerate(sentences):
        role = _sentence_role(sent, i, total)
        if role == "bg":
            bg_sents.append(sent)
        else:
            nov_sents.append(sent)

    # Ensure both have content
    if not bg_sents:
        bg_sents = [sentences[0]]
    if not nov_sents:
        nov_sents = [sentences[-1]]

    prefix = f"{title}. " if title else ""
    bg_text = prefix + " ".join(bg_sents)
    nov_text = prefix + " ".join(nov_sents)

    return bg_text, nov_text


# ── SPECTER2 encoding ────────────────────────────────────────────

def load_model(gpu_id: int = 0):
    """Load SPECTER2 model with proximity adapter."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    from adapters import AutoAdapterModel

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    log.info("Loading SPECTER2 + PRX adapter on %s", device)

    model_name = "allenai/specter2_base"
    adapter_name = "allenai/specter2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        model = AutoAdapterModel.from_pretrained(model_name)
        model.load_adapter(adapter_name, source="hf", set_active=True)
    except Exception:
        log.warning("adapter-transformers failed, trying base model")
        model = AutoModel.from_pretrained(model_name)

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode texts to embeddings."""
    import torch

    all_embs = []
    n = len(texts)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            token_embs = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embs = (token_embs * mask).sum(dim=1) / mask.sum(dim=1)

        all_embs.append(embs.cpu().numpy().astype(np.float32))

        if (start // batch_size) % 50 == 0:
            log.info("  Encoded %d/%d (%.1f%%)", end, n, end / n * 100)

    return np.vstack(all_embs)


# ── k-NN construction ────────────────────────────────────────────

def build_knn_and_save(
    emb: np.ndarray,
    work_ids: list[str],
    k: int,
    out_path: Path,
    nthreads: int = 16,
):
    """Build k-NN edges and save as parquet."""
    n, d = emb.shape
    faiss.omp_set_num_threads(nthreads)

    # Normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = (emb / norms).astype(np.float32)

    # IVF for large N, Flat for small
    if n > 200_000:
        nlist = int(np.sqrt(n)) * 2
        nprobe = min(nlist // 16, 64)
        log.info("  IVF index: nlist=%d, nprobe=%d", nlist, nprobe)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = nprobe
        index.train(emb_norm)
        index.add(emb_norm)
    else:
        index = faiss.IndexFlatIP(d)
        index.add(emb_norm)

    t0 = time.time()
    sims, idxs = index.search(emb_norm, k + 1)
    log.info("  Search done in %.1fs", time.time() - t0)

    # Remove self (vectorized)
    row_ids = np.arange(n).reshape(-1, 1)
    self_mask = (idxs == row_ids)
    sims[self_mask] = -np.inf
    idxs[self_mask] = -1

    sort_idx = np.argsort(-sims, axis=1)[:, :k]
    ri = np.arange(n).reshape(-1, 1)
    dst = idxs[ri, sort_idx].ravel()
    sim = sims[ri, sort_idx].ravel().astype(np.float32)
    src = np.repeat(np.arange(n), k)
    valid = dst >= 0
    src, dst, sim = src[valid], dst[valid], sim[valid]

    # Symmetrize
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
    log.info("  Saved %d edges to %s", df.height, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=int, required=True, choices=[12, 15])
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gcc-mapping", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="/tmp/emb_bgnov_output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GCC node IDs
    gcc_ids = set(pl.read_parquet(args.gcc_mapping)["work_id"].to_list())
    log.info("GCC nodes: %d", len(gcc_ids))

    # Load docs
    docs_path = DOCS_DIR / DOCS_FILES[args.field]
    log.info("Loading docs from %s", docs_path)
    docs = pl.read_parquet(docs_path, columns=["work_id", "title", "abstract"])
    # Deduplicate and filter to GCC
    docs = docs.unique(subset=["work_id"]).filter(
        pl.col("work_id").is_in(pl.Series(sorted(gcc_ids)))
    )
    log.info("Docs after GCC filter: %d", docs.height)

    # Split bg/nov
    log.info("Splitting bg/nov sentences...")
    t0 = time.time()
    bg_texts = []
    nov_texts = []
    work_ids = []
    for row in docs.iter_rows(named=True):
        bg, nov = split_bg_nov(row["title"] or "", row["abstract"] or "")
        bg_texts.append(bg)
        nov_texts.append(nov)
        work_ids.append(row["work_id"])
    log.info("Split done in %.1fs: %d papers", time.time() - t0, len(work_ids))

    # Load model
    tokenizer, model, device = load_model(args.gpu)

    # Encode bg
    log.info("=== Encoding BG texts (%d) ===", len(bg_texts))
    t0 = time.time()
    emb_bg = encode_texts(bg_texts, tokenizer, model, device, args.batch_size)
    log.info("BG encoding done in %.1fs, shape=%s", time.time() - t0, emb_bg.shape)

    # Encode nov
    log.info("=== Encoding NOV texts (%d) ===", len(nov_texts))
    t0 = time.time()
    emb_nov = encode_texts(nov_texts, tokenizer, model, device, args.batch_size)
    log.info("NOV encoding done in %.1fs, shape=%s", time.time() - t0, emb_nov.shape)

    # Save raw embeddings
    with h5py.File(out_dir / f"emb_bg_field{args.field}.h5", "w") as f:
        f.create_dataset("embeddings", data=emb_bg)
    with h5py.File(out_dir / f"emb_nov_field{args.field}.h5", "w") as f:
        f.create_dataset("embeddings", data=emb_nov)
    pl.DataFrame({"work_id": work_ids}).write_parquet(
        out_dir / f"mapping_field{args.field}.parquet"
    )
    log.info("Saved raw embeddings")

    # Build k-NN for bg
    log.info("=== Building BG k-NN ===")
    build_knn_and_save(
        emb_bg, work_ids, args.k,
        out_dir / f"emb_bg_knn{args.k}_field{args.field}.parquet",
    )

    # Build k-NN for nov
    log.info("=== Building NOV k-NN ===")
    build_knn_and_save(
        emb_nov, work_ids, args.k,
        out_dir / f"emb_nov_knn{args.k}_field{args.field}.parquet",
    )

    log.info("=== All done! ===")


if __name__ == "__main__":
    main()
