"""Build citation edge tables from OpenAlex referenced_works.

Converts citing → [cited] maps into DC (direct citation) and
BC (bibliographic coupling) edge tables in sciscape format.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from .client import WorkRecord

log = logging.getLogger(__name__)


def build_citation_edges(
    works: Sequence[WorkRecord],
    *,
    normalization: str = "fractional",
    bc: bool = True,
    cc: bool = True,
    bc_topk: int = 50,
    min_shared_refs: int = 1,
) -> Dict[str, pl.DataFrame]:
    """Build DC and BC edge tables from OpenAlex work records.

    Parameters
    ----------
    works : list of WorkRecord
        Works with ``referenced_works`` populated.
    normalization : str
        DC normalization: "binary" or "fractional" (Waltman & Van Eck 2012).
    bc : bool
        Whether to compute bibliographic coupling edges.
    bc_topk : int
        Keep top-k BC neighbors per work.
    min_shared_refs : int
        Minimum shared references for a BC edge.

    Returns
    -------
    dict of str → pl.DataFrame
        Keys: "dc" (direct citation), "bc" (bibliographic coupling).
        Each DataFrame has columns: uid1, uid2, rel_sum2.
    """
    focal_ids = {w.id for w in works}
    id_to_idx = {w.id: i for i, w in enumerate(works)}
    n = len(works)

    result: Dict[str, pl.DataFrame] = {}

    # ── DC: direct citation within focal set ──────────────────
    dc_uid1, dc_uid2, dc_weight = [], [], []
    for w in works:
        refs = w.referenced_works or []
        n_refs = len(refs)
        wt = 1.0 / n_refs if normalization == "fractional" and n_refs > 0 else 1.0
        wid = w.id
        for ref_id in refs:
            if ref_id in focal_ids and ref_id != wid:  # exclude self-citation
                dc_uid1.append(wid)
                dc_uid2.append(ref_id)
                dc_weight.append(wt)

    if dc_uid1:
        dc_df = pl.DataFrame({"uid1": dc_uid1, "uid2": dc_uid2, "rel_sum2": dc_weight})
        # Symmetrize: add reverse direction (swap uid1 ↔ uid2)
        dc_rev = dc_df.select(
            pl.col("uid2").alias("uid1"),
            pl.col("uid1").alias("uid2"),
            pl.col("rel_sum2"),
        )
        dc_sym = pl.concat([dc_df, dc_rev]).group_by(["uid1", "uid2"]).agg(
            pl.col("rel_sum2").sum()
        )
        result["dc"] = dc_sym
        log.info("DC edges: %d (from %d works)", dc_sym.height, n)
    else:
        result["dc"] = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
        log.info("DC edges: 0 (no internal citations)")

    # ── BC: bibliographic coupling ────────────────────────────
    if bc:
        # Build reference matrix: focal works × all referenced works
        all_refs = set()
        for w in works:
            all_refs.update(w.referenced_works or [])
        ref_list = sorted(all_refs)
        ref_to_col = {r: j for j, r in enumerate(ref_list)}
        n_refs_total = len(ref_list)

        if n_refs_total > 0:
            # Sparse matrix: works × references (vectorized COO construction)
            # Pre-compute per-work ref counts and column indices in batch
            _rows, _cols, _data = [], [], []
            for i, w in enumerate(works):
                refs = w.referenced_works or []
                n_r = len(refs)
                if n_r == 0:
                    continue
                wt = 1.0 / n_r if normalization == "fractional" else 1.0
                mapped = [ref_to_col[r] for r in refs if r in ref_to_col]
                if mapped:
                    _rows.append(np.full(len(mapped), i, dtype=np.int32))
                    _cols.append(np.array(mapped, dtype=np.int32))
                    _data.append(np.full(len(mapped), wt))

            if _rows:
                row_idx = np.concatenate(_rows)
                col_idx = np.concatenate(_cols)
                data = np.concatenate(_data)
            else:
                row_idx = np.array([], dtype=np.int32)
                col_idx = np.array([], dtype=np.int32)
                data = np.array([], dtype=np.float64)

            M = csr_matrix(
                (data, (row_idx, col_idx)),
                shape=(n, n_refs_total),
            )
            # BC = M @ M^T
            BC = (M @ M.T).tocsr()

            # Pre-compute binary count matrix (once, outside loop)
            BC_count = None
            if min_shared_refs > 1:
                M_bin = M.copy()
                M_bin.data[:] = 1.0
                BC_count = (M_bin @ M_bin.T).tocsr()

            # Extract top-k edges (vectorized: collect arrays, not dict list)
            work_ids = [w.id for w in works]  # pre-map index → UID
            bc_uid1, bc_uid2, bc_val = [], [], []
            for i in range(n):
                start, end = BC.indptr[i], BC.indptr[i + 1]
                if start == end:
                    continue
                cols = BC.indices[start:end]
                vals = BC.data[start:end]

                # Filter: j > i (upper triangle), min shared refs
                mask = cols > i
                cols_f = cols[mask]
                vals_f = vals[mask]

                if BC_count is not None:
                    cnt_start = BC_count.indptr[i]
                    cnt_end = BC_count.indptr[i + 1]
                    cnt_cols = BC_count.indices[cnt_start:cnt_end]
                    cnt_data = BC_count.data[cnt_start:cnt_end]
                    if len(cnt_cols) > 0:
                        cnt_idx = np.searchsorted(cnt_cols, cols_f)
                        cnt_idx_safe = np.clip(cnt_idx, 0, len(cnt_cols) - 1)
                        cnt_valid = (cnt_idx < len(cnt_cols)) & (cnt_cols[cnt_idx_safe] == cols_f)
                        counts = np.where(cnt_valid, cnt_data[cnt_idx_safe], 0)
                    else:
                        counts = np.zeros(len(cols_f), dtype=np.int32)
                    keep = counts >= min_shared_refs
                    cols_f = cols_f[keep]
                    vals_f = vals_f[keep]

                if len(cols_f) == 0:
                    continue

                # Top-k
                if len(cols_f) > bc_topk:
                    topk_idx = np.argpartition(-vals_f, bc_topk)[:bc_topk]
                    cols_f = cols_f[topk_idx]
                    vals_f = vals_f[topk_idx]

                # Batch append (avoid per-edge dict creation)
                wid = work_ids[i]
                bc_uid1.extend([wid] * len(cols_f))
                bc_uid2.extend(work_ids[j] for j in cols_f)
                bc_val.extend(vals_f.tolist())

            if bc_uid1:
                bc_df = pl.DataFrame({"uid1": bc_uid1, "uid2": bc_uid2, "rel_sum2": bc_val})
                # Symmetrize
                bc_rev = bc_df.select(
                    pl.col("uid2").alias("uid1"),
                    pl.col("uid1").alias("uid2"),
                    pl.col("rel_sum2"),
                )
                bc_sym = pl.concat([bc_df, bc_rev]).group_by(["uid1", "uid2"]).agg(
                    pl.col("rel_sum2").sum()
                )
                result["bc"] = bc_sym
                log.info("BC edges: %d", bc_sym.height)
            else:
                result["bc"] = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
                log.info("BC edges: 0")
        else:
            result["bc"] = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
            log.info("BC edges: 0 (no references)")

    # ── CC: co-citation (from referenced_works, approximate) ──
    if cc:
        # CC approximation: two focal papers are "co-cited" if they share
        # a common citer within the focal set.
        # Build citing matrix: for each work, who cites it?
        # cited_by[ref_id] = list of focal works that cite ref_id
        cited_by: Dict[str, List[int]] = {}
        for i, w in enumerate(works):
            for ref_id in (w.referenced_works or []):
                if ref_id in focal_ids:
                    cited_by.setdefault(ref_id, []).append(i)

        # CC: two focal works (A, B) are co-cited if a third focal work C
        # cites both A and B (i.e., both A and B are in C's reference list)
        # Equivalently: for each focal work C, all pairs in C's references
        # that are also focal → co-citation edge
        cc_weights: Dict[tuple, float] = {}
        for w in works:
            focal_refs = [r for r in (w.referenced_works or []) if r in focal_ids]
            n_r = len(focal_refs)
            if n_r < 2:
                continue
            w_frac = 1.0 / (n_r * (n_r - 1) / 2) if normalization == "fractional" else 1.0
            focal_refs_sorted = sorted(focal_refs)
            for a, b in combinations(focal_refs_sorted, 2):
                cc_weights[(a, b)] = cc_weights.get((a, b), 0) + w_frac

        if cc_weights:
            cc_pairs = list(cc_weights.keys())
            cc_df = pl.DataFrame({
                "uid1": [p[0] for p in cc_pairs],
                "uid2": [p[1] for p in cc_pairs],
                "rel_sum2": [cc_weights[p] for p in cc_pairs],
            })
            cc_rev = cc_df.select(
                pl.col("uid2").alias("uid1"),
                pl.col("uid1").alias("uid2"),
                pl.col("rel_sum2"),
            )
            cc_sym = pl.concat([cc_df, cc_rev]).group_by(["uid1", "uid2"]).agg(
                pl.col("rel_sum2").sum()
            )
            result["cc"] = cc_sym
            log.info("CC edges: %d", cc_sym.height)
        else:
            result["cc"] = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
            log.info("CC edges: 0")

    return result


def works_to_abstracts(works: Sequence[WorkRecord]) -> pl.DataFrame:
    """Convert WorkRecords to abstracts DataFrame (sciscape format)."""
    return pl.DataFrame({
        "uid": [w.id for w in works],
        "title": [w.title for w in works],
        "abstract": [w.abstract for w in works],
        "pubyear": [w.year for w in works],
    })


__all__ = ["build_citation_edges", "works_to_abstracts"]
