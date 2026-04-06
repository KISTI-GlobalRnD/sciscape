"""Build DC, BC, CC edge tables from raw citation data.

Mathematical definitions
------------------------
Given a citation table (citing_id, cited_id):

- **DC** (Direct Citation):  A cites B → edge(A, B).
  - binary: weight = 1 per citation
  - fractional: weight = 1 / total_references(A)  (Waltman & Van Eck 2012)

- **BC** (Bibliographic Coupling):  A and B share references.
  - M[i, j] = 1 if paper *i* cites reference *j*  (j can be ANY paper)
  - BC = M @ M^T  → BC[i, j] = number of shared references

- **CC** (Co-Citation):  A and B are cited together.
  - M[i, j] = 1 if citer *i* cites paper *j*  (i can be ANY paper)
  - CC = M^T @ M  → CC[i, j] = number of shared citers

Normalization variants (BC/CC):
  - raw:             w(i, j) = shared count
  - cosine:          w(i, j) = shared / sqrt(deg_i * deg_j)
  - assoc_strength:  w(i, j) = shared / (deg_i * deg_j)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
import polars as pl
from scipy import sparse

from .config import DCNormalization, LinkageConfig, Normalization

log = logging.getLogger(__name__)

try:
    from sparse_dot_topn import sp_matmul_topn as _sp_matmul_topn

    _HAS_SPARSE_DOT_TOPN = True
except ImportError:
    _HAS_SPARSE_DOT_TOPN = False


# ── Internal types ────────────────────────────────────────────────
# (rows, cols, weights) — COO-style arrays for a single edge set
_EdgeArrays = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _matmul_topk(
    A: sparse.spmatrix,
    B: sparse.spmatrix,
    top_k: int | None,
) -> sparse.csr_matrix:
    """Fused sparse matmul + per-row top-k.

    When ``sparse_dot_topn`` is installed and *top_k* is set, the full
    result matrix is never materialised — only the top-k entries per row
    are kept during the multiply, which is critical for large-scale BC/CC
    where ``M @ M.T`` would otherwise explode in memory.

    Falls back to plain ``(A @ B).tocsr()`` when the library is absent or
    *top_k* is None.
    """
    if top_k is not None and _HAS_SPARSE_DOT_TOPN:
        log.info("_matmul_topk: using sparse_dot_topn (top_k=%d)", top_k)
        return _sp_matmul_topn(
            A.tocsr(), B.tocsc(),
            top_n=top_k,
            sort=True,
        ).tocsr()

    if top_k is not None and not _HAS_SPARSE_DOT_TOPN:
        log.warning(
            "_matmul_topk: sparse_dot_topn not installed, falling back to "
            "dense matmul. Install with: pip install 'sciscape[largescale]'"
        )
    return (A @ B).tocsr()


# ═══════════════════════════════════════════════════════════════════
# Node-set helpers (vectorized — no Python dicts)
# ═══════════════════════════════════════════════════════════════════

def _build_uid_map(
    node_ids: Set[str],
) -> Tuple[pl.Series, pl.DataFrame]:
    """Deterministic sorted mapping: work_id → integer index (Polars-native).

    Returns
    -------
    categories : pl.Series
        Sorted unique UIDs (for ``Series.gather()`` index→string lookup).
    uid_map : pl.DataFrame
        Columns ``uid``, ``_idx`` — for join-based string→index lookup.
    """
    categories = pl.Series("uid", sorted(node_ids))
    n = categories.len()
    uid_map = pl.DataFrame({
        "uid": categories,
        "_idx": np.arange(n, dtype=np.int32),
    })
    return categories, uid_map


def _edge_arrays_to_dataframe(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    categories: pl.Series,
) -> pl.DataFrame:
    """Convert COO arrays to a Polars edge DataFrame (uid1, uid2, rel_sum2)."""
    return pl.DataFrame({
        "uid1": categories.gather(rows),
        "uid2": categories.gather(cols),
        "rel_sum2": weights,
    })


# ═══════════════════════════════════════════════════════════════════
# DC: Direct Citation
# ═══════════════════════════════════════════════════════════════════

def build_dc(
    citations: pl.DataFrame,
    node_ids: Set[str],
    *,
    config: Optional[LinkageConfig] = None,
    norms: Optional[Sequence[DCNormalization]] = None,
) -> Dict[str, pl.DataFrame]:
    """Build direct-citation edge tables.

    Parameters
    ----------
    citations : pl.DataFrame
        Raw citation table with at least (citing_col, cited_col) columns.
    node_ids : set of str
        Focal node IDs (e.g. GCC+k30 papers).
    config : LinkageConfig, optional
        Configuration (column names, etc.).  Defaults are used if None.
    norms : sequence of DCNormalization, optional
        Override which normalizations to compute.

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping from name (e.g. ``"dc_binary"``) to edge DataFrame
        with columns ``uid1, uid2, rel_sum2``.
    """
    cfg = config or LinkageConfig()
    norms = norms or cfg.dc_norms
    citing_col, cited_col = cfg.citing_col, cfg.cited_col

    categories, uid_map = _build_uid_map(node_ids)

    # Filter: both ends in node set
    dc = citations.filter(
        pl.col(citing_col).is_in(node_ids)
        & pl.col(cited_col).is_in(node_ids)
    )
    if cfg.cited_in_set_col and cfg.cited_in_set_col in citations.columns:
        dc = dc.filter(pl.col(cfg.cited_in_set_col) == 1)

    log.info("DC: %d directed citations among %d nodes", dc.height, len(node_ids))

    if dc.height == 0:
        log.warning("DC: no edges produced")
        empty = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []}).cast(
            {"uid1": pl.Utf8, "uid2": pl.Utf8, "rel_sum2": pl.Float64}
        )
        return {f"dc_{n.value}": empty for n in norms}

    # ── Vectorized UID → int mapping via Polars join ──────────────
    dc_idx = (
        dc.select(pl.col(citing_col), pl.col(cited_col))
        .join(
            uid_map.rename({"uid": citing_col, "_idx": "_ai"}),
            on=citing_col, how="left",
        )
        .join(
            uid_map.rename({"uid": cited_col, "_idx": "_bi"}),
            on=cited_col, how="left",
        )
        .with_columns(
            pl.min_horizontal("_ai", "_bi").alias("_lo"),
            pl.max_horizontal("_ai", "_bi").alias("_hi"),
        )
    )

    result = {}

    if DCNormalization.BINARY in norms:
        agg = dc_idx.group_by("_lo", "_hi").len()
        rows = agg["_lo"].to_numpy()
        cols = agg["_hi"].to_numpy()
        w = agg["len"].to_numpy().astype(np.float32)
        name = "dc_binary"
        result[name] = _edge_arrays_to_dataframe(rows, cols, w, categories)
        log.info("  %s: mean=%.6f, max=%.4f", name, w.mean(), w.max())

    if DCNormalization.FRACTIONAL in norms:
        # Reference counts per citing paper (ALL references, not just in-set)
        rc_df = (
            citations.filter(pl.col(citing_col).is_in(node_ids))
            .group_by(citing_col).len()
            .rename({"len": "_ref_count"})
        )
        dc_frac = (
            dc_idx.join(rc_df, on=citing_col, how="left")
            .with_columns(
                (1.0 / pl.col("_ref_count").fill_null(1).cast(pl.Float64))
                .alias("_frac_w")
            )
        )
        agg = dc_frac.group_by("_lo", "_hi").agg(
            pl.col("_frac_w").sum()
        )
        rows = agg["_lo"].to_numpy()
        cols = agg["_hi"].to_numpy()
        w = agg["_frac_w"].to_numpy().astype(np.float32)
        name = "dc_fractional"
        result[name] = _edge_arrays_to_dataframe(rows, cols, w, categories)
        log.info("  %s: mean=%.6f, max=%.4f", name, w.mean(), w.max())

    n_edges = next(iter(result.values())).height if result else 0
    log.info("DC: %d undirected edges", n_edges)

    return result


# ═══════════════════════════════════════════════════════════════════
# BC: Bibliographic Coupling (shared references)
# ═══════════════════════════════════════════════════════════════════

def build_bc(
    citations: pl.DataFrame,
    node_ids: Set[str],
    *,
    config: Optional[LinkageConfig] = None,
    norms: Optional[Sequence[Normalization]] = None,
) -> Dict[str, pl.DataFrame]:
    """Build bibliographic-coupling edge tables.

    BC uses ALL references from focal papers (including out-of-field targets)
    to construct the paper-reference bipartite matrix.

    Parameters
    ----------
    citations : pl.DataFrame
        Raw citation table.
    node_ids : set of str
        Focal node IDs.

    Returns
    -------
    dict[str, pl.DataFrame]
        ``"bc_raw"``, ``"bc_cosine"``, ``"bc_assoc_strength"`` → edge DataFrames.
    """
    cfg = config or LinkageConfig()
    norms = norms or cfg.bc_norms
    citing_col, cited_col = cfg.citing_col, cfg.cited_col

    categories, uid_map = _build_uid_map(node_ids)
    n = categories.len()

    # All references FROM focal papers (to ANY paper, including out-of-field)
    refs = citations.filter(pl.col(citing_col).is_in(node_ids))
    log.info("BC: %d references from %d focal papers",
             refs.height, refs[citing_col].n_unique())

    # ── Vectorized mapping via Polars join ────────────────────────
    # Row indices: focal paper → node index
    row_idx = (
        refs.select(pl.col(citing_col).alias("uid"))
        .join(uid_map, on="uid", how="left")["_idx"]
        .to_numpy().astype(np.int32)
    )

    # Column indices: cited paper → reference universe index
    cited_uids = refs[cited_col].unique().sort()
    n_refs = cited_uids.len()
    cited_map = pl.DataFrame({
        "uid": cited_uids,
        "_idx": np.arange(n_refs, dtype=np.int32),
    })
    col_idx = (
        refs.select(pl.col(cited_col).alias("uid"))
        .join(cited_map, on="uid", how="left")["_idx"]
        .to_numpy().astype(np.int32)
    )

    data = np.ones(len(row_idx), dtype=np.float32)

    M = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n_refs))
    log.info("BC: bipartite matrix %d x %d, nnz=%d", n, n_refs, M.nnz)

    # BC = M @ M^T (fused top-k when sparse_dot_topn is available)
    t0 = time.time()
    BC = _matmul_topk(M, M.T, top_k=cfg.bc_topk)
    BC.setdiag(0)
    BC.eliminate_zeros()
    log.info("BC: M@M^T done in %.1fs, nnz=%d", time.time() - t0, BC.nnz)

    # Min-shared filter (BC uses stricter threshold by default)
    min_s = cfg.bc_min_shared
    if min_s > 1:
        BC.data[BC.data < min_s] = 0
        BC.eliminate_zeros()
        log.info("BC: after min_shared>=%d, nnz=%d", min_s, BC.nnz)

    return _normalize_coupling_matrix(
        BC, M, categories, norms, prefix="bc", axis=1,
    )


# ═══════════════════════════════════════════════════════════════════
# CC: Co-Citation (shared citers)
# ═══════════════════════════════════════════════════════════════════

def build_cc(
    citations: pl.DataFrame,
    node_ids: Set[str],
    *,
    config: Optional[LinkageConfig] = None,
    norms: Optional[Sequence[Normalization]] = None,
) -> Dict[str, pl.DataFrame]:
    """Build co-citation edge tables.

    CC uses citations TO focal papers from any paper (citer need not be focal).

    Parameters
    ----------
    citations : pl.DataFrame
        Raw citation table.
    node_ids : set of str
        Focal node IDs.

    Returns
    -------
    dict[str, pl.DataFrame]
        ``"cc_raw"``, ``"cc_cosine"``, ``"cc_assoc_strength"`` → edge DataFrames.
    """
    cfg = config or LinkageConfig()
    norms = norms or cfg.cc_norms
    citing_col, cited_col = cfg.citing_col, cfg.cited_col

    categories, uid_map = _build_uid_map(node_ids)
    n = categories.len()

    # Citations TO focal papers (citers can be any paper)
    cc_cit = citations.filter(pl.col(cited_col).is_in(node_ids))
    if cfg.cited_in_set_col and cfg.cited_in_set_col in citations.columns:
        cc_cit = cc_cit.filter(pl.col(cfg.cited_in_set_col) == 1)

    log.info("CC: %d citations to %d focal papers from %d citers",
             cc_cit.height,
             cc_cit[cited_col].n_unique(),
             cc_cit[citing_col].n_unique())

    # ── Vectorized mapping via Polars join ────────────────────────
    # Row indices: citer → citer universe index
    citer_uids = cc_cit[citing_col].unique().sort()
    n_citers = citer_uids.len()
    citer_map = pl.DataFrame({
        "uid": citer_uids,
        "_idx": np.arange(n_citers, dtype=np.int32),
    })

    row_idx = (
        cc_cit.select(pl.col(citing_col).alias("uid"))
        .join(citer_map, on="uid", how="left")["_idx"]
        .to_numpy().astype(np.int32)
    )

    # Column indices: cited focal paper → node index
    col_idx = (
        cc_cit.select(pl.col(cited_col).alias("uid"))
        .join(uid_map, on="uid", how="left")["_idx"]
        .to_numpy().astype(np.int32)
    )

    data = np.ones(len(row_idx), dtype=np.float32)

    M = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_citers, n))
    log.info("CC: bipartite matrix %d x %d, nnz=%d", n_citers, n, M.nnz)

    # CC = M^T @ M (fused top-k when sparse_dot_topn is available)
    t0 = time.time()
    CC = _matmul_topk(M.T, M, top_k=cfg.cc_topk)
    CC.setdiag(0)
    CC.eliminate_zeros()
    log.info("CC: M^T@M done in %.1fs, nnz=%d", time.time() - t0, CC.nnz)

    # Min-shared filter
    min_s = cfg.cc_min_shared
    if min_s > 1:
        CC.data[CC.data < min_s] = 0
        CC.eliminate_zeros()
        log.info("CC: after min_shared>=%d, nnz=%d", min_s, CC.nnz)

    return _normalize_coupling_matrix(
        CC, M, categories, norms, prefix="cc", axis=0,
    )


# ═══════════════════════════════════════════════════════════════════
# Shared normalization helper
# ═══════════════════════════════════════════════════════════════════

def _normalize_coupling_matrix(
    coupling: sparse.csr_matrix,
    bipartite: sparse.csr_matrix,
    categories: pl.Series,
    norms: Sequence[Normalization],
    *,
    prefix: str,
    axis: int,
) -> Dict[str, pl.DataFrame]:
    """Extract upper triangle and apply normalizations.

    Parameters
    ----------
    coupling : sparse matrix
        n x n coupling matrix (BC or CC).
    bipartite : sparse matrix
        The bipartite matrix M used to compute coupling.
    categories : pl.Series
        Sorted unique UIDs (for ``Series.gather()`` index→string lookup).
    norms : sequence of Normalization
        Which normalizations to produce.
    prefix : str
        Name prefix ("bc" or "cc").
    axis : int
        Which axis of *bipartite* gives per-node degree.
        - BC: axis=1 (row sums = reference count per paper)
        - CC: axis=0 (column sums = citer count per paper)
    """
    upper = sparse.triu(coupling, k=1).tocoo()
    rows, cols = upper.row, upper.col
    shared = upper.data.astype(np.float32)

    if len(shared) == 0:
        log.warning("%s: no edges after filtering", prefix.upper())
        empty = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []}).cast(
            {"uid1": pl.Utf8, "uid2": pl.Utf8, "rel_sum2": pl.Float64}
        )
        return {f"{prefix}_{n.value}": empty for n in norms}

    log.info("%s: %d undirected edges", prefix.upper(), len(rows))

    # Per-node degree in the bipartite matrix
    deg = np.asarray(bipartite.sum(axis=axis)).ravel()
    s_a = deg[rows]
    s_b = deg[cols]

    result = {}
    for norm in norms:
        if norm == Normalization.RAW:
            w = shared.copy()
        elif norm == Normalization.COSINE:
            w = shared / np.sqrt(s_a * s_b)
        elif norm == Normalization.ASSOC_STRENGTH:
            w = shared / (s_a * s_b)
        else:
            raise ValueError(f"Unknown normalization: {norm}")

        name = f"{prefix}_{norm.value}"
        result[name] = _edge_arrays_to_dataframe(rows, cols, w, categories)
        log.info("  %s: %d edges, mean=%.6f, max=%.4f", name, len(w), w.mean(), w.max())

    return result


__all__ = ["build_bc", "build_cc", "build_dc"]
