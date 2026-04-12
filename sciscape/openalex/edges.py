"""Build citation edge tables from OpenAlex referenced_works.

Converts citing → [cited] maps into DC (direct citation) and
BC (bibliographic coupling) edge tables in sciscape format.
"""

from __future__ import annotations

import logging
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
    dc_rows: List[dict] = []
    for w in works:
        n_refs = len(w.referenced_works) if w.referenced_works else 0
        for ref_id in (w.referenced_works or []):
            if ref_id in focal_ids:
                weight = 1.0
                if normalization == "fractional" and n_refs > 0:
                    weight = 1.0 / n_refs
                dc_rows.append({
                    "uid1": w.id,
                    "uid2": ref_id,
                    "rel_sum2": weight,
                })

    if dc_rows:
        dc_df = pl.DataFrame(dc_rows)
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
            # Sparse matrix: works × references
            row_idx, col_idx, data = [], [], []
            for i, w in enumerate(works):
                refs = w.referenced_works or []
                n_r = len(refs)
                for ref_id in refs:
                    if ref_id in ref_to_col:
                        row_idx.append(i)
                        col_idx.append(ref_to_col[ref_id])
                        # Fractional counting
                        data.append(1.0 / n_r if n_r > 0 else 1.0)

            M = csr_matrix(
                (data, (row_idx, col_idx)),
                shape=(n, n_refs_total),
            )
            # BC = M @ M^T
            BC = (M @ M.T).tocsr()

            # Extract top-k edges
            bc_rows: List[dict] = []
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

                if min_shared_refs > 1:
                    # Count shared refs (use binary matrix)
                    M_bin = M.copy()
                    M_bin.data[:] = 1.0
                    BC_count = (M_bin @ M_bin.T).tocsr()
                    counts = np.array(BC_count[i, cols_f].todense()).ravel()
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

                for j_idx, (j, val) in enumerate(zip(cols_f, vals_f)):
                    bc_rows.append({
                        "uid1": works[i].id,
                        "uid2": works[j].id,
                        "rel_sum2": float(val),
                    })

            if bc_rows:
                bc_df = pl.DataFrame(bc_rows)
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
