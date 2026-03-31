#!/usr/bin/env python3
"""Build DC, BC, CC edge lists from raw oa26 citation data.

For each field (GCC+k30 node set), computes:
- DC:    direct citation (binary / fractional)
- BC:    bibliographic coupling (raw / cosine / assoc_strength)
- CC:    co-citation (raw / cosine / assoc_strength)
- ExtDC: 2-hop indirect citation (fractional)

Output: parquet edge lists per (field, link_type, normalization).

Usage:
    python scripts/build_linktype_edges.py --field 34
    python scripts/build_linktype_edges.py --field 34 --link-type bc --norm cosine
    python scripts/build_linktype_edges.py --field all
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build")

# ── Paths ──────────────────────────────────────────────────────────
DATA_ROOT = Path.home() / "Desktop/HDD/local_map_analysis_data/processed/outputs/data"
CITATION_DIR = DATA_ROOT / "oa26_citation_edges"
NODE_DIR = DATA_ROOT / "oa26_gcc_only_k30"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges"

SAMPLE_FIELDS = [34, 30, 29, 12]

# ── Filters ────────────────────────────────────────────────────────
MIN_SHARED = 2  # minimum shared refs/citers for BC/CC (standard practice)


def _update_min_shared(val: int):
    global MIN_SHARED
    MIN_SHARED = val


def load_nodes(field_id: int) -> set[str]:
    """Load GCC+k30 work_ids for a field."""
    path = NODE_DIR / f"field_{field_id}_nodes_oa_meta_gcc_k30.parquet"
    df = pl.read_parquet(path, columns=["work_id"])
    ids = set(df["work_id"].unique().to_list())
    log.info("Field %d: %d unique GCC+k30 nodes", field_id, len(ids))
    return ids


def load_citations(field_id: int) -> pl.DataFrame:
    """Load raw citation edges for a field."""
    path = CITATION_DIR / f"oa26_citation_edges_field_{field_id}.parquet"
    df = pl.read_parquet(path, columns=[
        "citing_work_id", "cited_work_id", "cited_in_set",
    ])
    log.info("Field %d: %d raw citations", field_id, df.shape[0])
    return df


# ═══════════════════════════════════════════════════════════════════
# DC: Direct Citation
# ═══════════════════════════════════════════════════════════════════

def build_dc(cit: pl.DataFrame, node_ids: set[str], id2idx: dict[str, int]):
    """Build DC edges. Returns dict of {norm_name: (rows, cols, weights)}."""
    # Filter: both ends in GCC+k30, cited_in_set=1
    dc = cit.filter(
        (pl.col("cited_in_set") == 1)
        & pl.col("citing_work_id").is_in(node_ids)
        & pl.col("cited_work_id").is_in(node_ids)
    )
    log.info("  DC raw edges: %d", dc.shape[0])

    citing = dc["citing_work_id"].to_list()
    cited = dc["cited_work_id"].to_list()

    # Reference counts per citing paper (for fractional normalization)
    # Use ALL references, not just in-set (matches Waltman & Van Eck 2012)
    ref_counts_df = cit.filter(
        pl.col("citing_work_id").is_in(node_ids)
    ).group_by("citing_work_id").len()
    ref_counts = dict(zip(
        ref_counts_df["citing_work_id"].to_list(),
        ref_counts_df["len"].to_list(),
    ))

    # Make undirected: for each A→B, add edge (min(A,B), max(A,B))
    edges: dict[tuple[int, int], dict] = {}
    for a_id, b_id in zip(citing, cited):
        ai, bi = id2idx[a_id], id2idx[b_id]
        key = (min(ai, bi), max(ai, bi))
        if key not in edges:
            edges[key] = {"binary": 0.0, "fractional": 0.0}
        edges[key]["binary"] += 1.0
        ref_a = ref_counts.get(a_id, 1)
        edges[key]["fractional"] += 1.0 / ref_a

    rows = np.array([k[0] for k in edges])
    cols = np.array([k[1] for k in edges])
    w_binary = np.array([v["binary"] for v in edges.values()])
    w_frac = np.array([v["fractional"] for v in edges.values()])

    log.info("  DC undirected edges: %d", len(edges))
    return {
        "dc_binary": (rows, cols, w_binary),
        "dc_fractional": (rows, cols, w_frac),
    }


# ═══════════════════════════════════════════════════════════════════
# BC: Bibliographic Coupling (shared references)
# ═══════════════════════════════════════════════════════════════════

def build_bc(cit: pl.DataFrame, node_ids: set[str], id2idx: dict[str, int]):
    """Build BC via sparse matrix multiply. Returns dict of norm→(r,c,w)."""
    n = len(id2idx)

    # All references FROM GCC+k30 papers (to ANY paper, including out-of-field)
    refs = cit.filter(pl.col("citing_work_id").is_in(node_ids))
    log.info("  BC: %d references from %d papers", refs.shape[0], refs["citing_work_id"].n_unique())

    # Map cited papers to column indices
    cited_ids = refs["cited_work_id"].unique().to_list()
    cited2col = {cid: i for i, cid in enumerate(cited_ids)}
    n_refs = len(cited2col)

    # Build sparse matrix: paper × reference (binary)
    citing_list = refs["citing_work_id"].to_list()
    cited_list = refs["cited_work_id"].to_list()

    row_idx = np.array([id2idx[c] for c in citing_list])
    col_idx = np.array([cited2col[c] for c in cited_list])
    data = np.ones(len(row_idx), dtype=np.float32)

    M = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n_refs))
    log.info("  BC: sparse matrix %d × %d, nnz=%d", n, n_refs, M.nnz)

    # BC = M @ M.T → shared reference counts
    t0 = time.time()
    BC = (M @ M.T).tocsr()
    BC.setdiag(0)  # remove self-loops
    BC.eliminate_zeros()
    log.info("  BC: M@M.T done in %.1fs, nnz=%d", time.time() - t0, BC.nnz)

    # Filter: min_shared >= MIN_SHARED
    BC.data[BC.data < MIN_SHARED] = 0
    BC.eliminate_zeros()
    log.info("  BC: after min_shared≥%d filter, nnz=%d", MIN_SHARED, BC.nnz)

    # Extract upper triangle only
    BC_upper = sparse.triu(BC, k=1).tocoo()
    rows, cols, shared = BC_upper.row, BC_upper.col, BC_upper.data.astype(np.float64)
    log.info("  BC: %d undirected edges", len(rows))

    # Reference count per paper (degree in M)
    ref_count = np.array(M.sum(axis=1)).ravel()  # shape (n,)

    # Normalizations
    s_a = ref_count[rows]
    s_b = ref_count[cols]

    w_raw = shared.copy()
    w_cosine = shared / np.sqrt(s_a * s_b)
    w_assoc = shared / (s_a * s_b)

    return {
        "bc_raw": (rows, cols, w_raw),
        "bc_cosine": (rows, cols, w_cosine),
        "bc_assoc_strength": (rows, cols, w_assoc),
    }


# ═══════════════════════════════════════════════════════════════════
# CC: Co-citation (shared citers)
# ═══════════════════════════════════════════════════════════════════

def build_cc(cit: pl.DataFrame, node_ids: set[str], id2idx: dict[str, int]):
    """Build CC via sparse matrix multiply. Returns dict of norm→(r,c,w)."""
    n = len(id2idx)

    # Citations TO GCC+k30 papers from any field paper
    # cited_in_set=1 means cited paper is in the field, but might not be in GCC+k30
    cc_cit = cit.filter(
        (pl.col("cited_in_set") == 1)
        & pl.col("cited_work_id").is_in(node_ids)
    )
    log.info("  CC: %d citations TO GCC+k30 papers from %d citers",
             cc_cit.shape[0], cc_cit["citing_work_id"].n_unique())

    # Map citers to row indices (citers may or may not be in GCC+k30)
    citer_ids = cc_cit["citing_work_id"].unique().to_list()
    citer2row = {cid: i for i, cid in enumerate(citer_ids)}
    n_citers = len(citer2row)

    citing_list = cc_cit["citing_work_id"].to_list()
    cited_list = cc_cit["cited_work_id"].to_list()

    # Build sparse matrix: citer × cited_paper (binary)
    row_idx = np.array([citer2row[c] for c in citing_list])
    col_idx = np.array([id2idx[c] for c in cited_list])
    data = np.ones(len(row_idx), dtype=np.float32)

    M = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_citers, n))
    log.info("  CC: sparse matrix %d × %d, nnz=%d", n_citers, n, M.nnz)

    # CC = M.T @ M → shared citer counts between GCC+k30 papers
    t0 = time.time()
    CC = (M.T @ M).tocsr()
    CC.setdiag(0)
    CC.eliminate_zeros()
    log.info("  CC: M.T@M done in %.1fs, nnz=%d", time.time() - t0, CC.nnz)

    # Filter: min_shared >= MIN_SHARED
    CC.data[CC.data < MIN_SHARED] = 0
    CC.eliminate_zeros()
    log.info("  CC: after min_shared≥%d filter, nnz=%d", MIN_SHARED, CC.nnz)

    # Upper triangle
    CC_upper = sparse.triu(CC, k=1).tocoo()
    rows, cols, shared = CC_upper.row, CC_upper.col, CC_upper.data.astype(np.float64)
    log.info("  CC: %d undirected edges", len(rows))

    # Citer count per paper (column sum of M)
    citer_count = np.array(M.sum(axis=0)).ravel()  # shape (n,)

    s_a = citer_count[rows]
    s_b = citer_count[cols]

    w_raw = shared.copy()
    w_cosine = shared / np.sqrt(s_a * s_b)
    w_assoc = shared / (s_a * s_b)

    return {
        "cc_raw": (rows, cols, w_raw),
        "cc_cosine": (rows, cols, w_cosine),
        "cc_assoc_strength": (rows, cols, w_assoc),
    }


# ═══════════════════════════════════════════════════════════════════
# ExtDC: Extended Direct Citation (Waltman et al. 2020)
#   Non-focal nodes expand DC coverage; Leiden uses node_sizes for
#   modified CPM where non-focal nodes incur no γ penalty.
#   Reference: QSS 1(2), 691-713, Appendix C.
# ═══════════════════════════════════════════════════════════════════

def build_ext_dc(cit: pl.DataFrame, node_ids: set[str], id2idx: dict[str, int]):
    """Build Waltman-style Extended DC with non-focal bridge nodes.

    Returns edge dict AND a separate focal_flags array (saved alongside edges).
    The extended graph includes:
      - focal nodes (GCC+k30): node_size = 1
      - non-focal nodes (cited/citing >=2 focal): node_size = 0
    Edges are fractional-weighted direct citations over the extended node set.
    """
    n_focal = len(id2idx)

    # All citations where at least one end is focal
    cit_focal = cit.filter(
        pl.col("citing_work_id").is_in(node_ids)
        | pl.col("cited_work_id").is_in(node_ids)
    )

    # ── Non-focal selection: papers with citation links to >=2 focal ──
    # Case 1: non-focal cites focal
    nf1 = cit_focal.filter(
        ~pl.col("citing_work_id").is_in(node_ids)
        & pl.col("cited_work_id").is_in(node_ids)
    ).select(
        pl.col("citing_work_id").alias("nf"),
        pl.col("cited_work_id").alias("focal"),
    )
    # Case 2: focal cites non-focal
    nf2 = cit_focal.filter(
        pl.col("citing_work_id").is_in(node_ids)
        & ~pl.col("cited_work_id").is_in(node_ids)
    ).select(
        pl.col("cited_work_id").alias("nf"),
        pl.col("citing_work_id").alias("focal"),
    )
    nf_counts = (
        pl.concat([nf1, nf2]).unique()
        .group_by("nf").agg(pl.col("focal").n_unique().alias("n_focal"))
    )
    non_focal_ids = set(
        nf_counts.filter(pl.col("n_focal") >= 2)["nf"].to_list()
    )
    log.info("  ExtDC: %d non-focal nodes (connected to >=2 focal)", len(non_focal_ids))

    # ── Extended node set: focal + non-focal ──
    all_ext = node_ids | non_focal_ids
    # Keep focal indices 0..n_focal-1 unchanged; non-focal start at n_focal
    nf_sorted = sorted(non_focal_ids)
    ext_id2idx = dict(id2idx)  # copy focal mapping
    for i, nf_id in enumerate(nf_sorted):
        ext_id2idx[nf_id] = n_focal + i
    n_total = n_focal + len(nf_sorted)
    ext_idx2id = [None] * n_total
    for wid, idx in ext_id2idx.items():
        ext_idx2id[idx] = wid

    # ── Build fractional-weighted edges over extended set ──
    ext_cit = cit_focal.filter(
        pl.col("citing_work_id").is_in(all_ext)
        & pl.col("cited_work_id").is_in(all_ext)
    )
    log.info("  ExtDC: %d directed citations in extended set", len(ext_cit))

    # Reference counts for fractional normalization (all refs of each paper)
    ref_counts = dict(
        cit.filter(pl.col("citing_work_id").is_in(all_ext))
        .group_by("citing_work_id").len().iter_rows()
    )

    edges: dict[tuple[int, int], float] = {}
    for a, b in ext_cit.select("citing_work_id", "cited_work_id").iter_rows():
        ai, bi = ext_id2idx[a], ext_id2idx[b]
        key = (min(ai, bi), max(ai, bi))
        edges[key] = edges.get(key, 0.0) + 1.0 / ref_counts.get(a, 1)

    rows = np.array([k[0] for k in edges])
    cols = np.array([k[1] for k in edges])
    weights = np.array([v for v in edges.values()])

    log.info("  ExtDC: %d undirected edges (%d focal + %d non-focal nodes)",
             len(edges), n_focal, len(nf_sorted))

    # Return edges AND metadata for save_ext_dc_edges
    return {
        "extdc_waltman": (rows, cols, weights),
        "_extdc_meta": (ext_idx2id, n_focal),
    }


# ═══════════════════════════════════════════════════════════════════
# Save / utility
# ═══════════════════════════════════════════════════════════════════

def save_edges(
    rows: np.ndarray, cols: np.ndarray, weights: np.ndarray,
    idx2id: list[str], field_id: int, name: str,
):
    """Save edge list as parquet."""
    out_dir = OUTPUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"

    df = pl.DataFrame({
        "src": [idx2id[i] for i in rows],
        "dst": [idx2id[i] for i in cols],
        "weight": weights,
    })
    df.write_parquet(path)
    log.info("  Saved %s: %d edges → %s", name, len(rows), path)
    return path


def print_summary(all_results: dict, field_id: int, n_nodes: int):
    """Print summary table of all built link types."""
    print(f"\n{'='*80}")
    print(f"FIELD {field_id}: {n_nodes} nodes — Link Type Edge Summary")
    print(f"{'='*80}")
    print(f"{'Link Type':<25} {'Edges':>10} {'Mean Weight':>12} {'Median':>10} {'Max':>10}")
    print("-" * 72)
    for name, (rows, cols, weights) in sorted(all_results.items()):
        print(f"{name:<25} {len(rows):>10,} {weights.mean():>12.6f} "
              f"{np.median(weights):>10.6f} {weights.max():>10.4f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def build_field(field_id: int, link_types: list[str] | None = None):
    """Build all link types for one field."""
    t_start = time.time()
    log.info("Building link types for field %d...", field_id)

    node_ids = load_nodes(field_id)
    cit = load_citations(field_id)

    # Stable ordering: sorted work_ids → index
    sorted_ids = sorted(node_ids)
    id2idx = {wid: i for i, wid in enumerate(sorted_ids)}
    idx2id = sorted_ids

    all_types = {"dc", "bc", "cc", "extdc"}
    if link_types is None:
        link_types = sorted(all_types)
    else:
        link_types = [lt.lower() for lt in link_types]

    all_results = {}

    if "dc" in link_types:
        log.info("── Building DC ──")
        all_results.update(build_dc(cit, node_ids, id2idx))

    if "bc" in link_types:
        log.info("── Building BC ──")
        all_results.update(build_bc(cit, node_ids, id2idx))

    if "cc" in link_types:
        log.info("── Building CC ──")
        all_results.update(build_cc(cit, node_ids, id2idx))

    if "extdc" in link_types:
        log.info("── Building Extended DC (Waltman) ──")
        extdc_result = build_ext_dc(cit, node_ids, id2idx)
        # Separate metadata from edge data
        ext_meta = extdc_result.pop("_extdc_meta")
        all_results.update(extdc_result)

    # Save all edge lists
    out_dir = OUTPUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (rows, cols, weights) in all_results.items():
        if name == "extdc_waltman":
            # ExtDC uses extended node set (focal + non-focal)
            ext_idx2id, n_focal = ext_meta
            save_edges(rows, cols, weights, ext_idx2id, field_id, name)
            # Save node info: work_id + is_focal flag
            node_df = pl.DataFrame({
                "idx": list(range(len(ext_idx2id))),
                "work_id": ext_idx2id,
                "is_focal": [1 if i < n_focal else 0
                             for i in range(len(ext_idx2id))],
            })
            node_path = out_dir / "extdc_waltman_nodes.parquet"
            node_df.write_parquet(node_path)
            log.info("  Saved ExtDC node info: %d focal + %d non-focal → %s",
                     n_focal, len(ext_idx2id) - n_focal, node_path)
        else:
            save_edges(rows, cols, weights, idx2id, field_id, name)

    # Save focal node mapping
    pl.DataFrame({"idx": list(range(len(idx2id))), "work_id": idx2id}).write_parquet(
        out_dir / "node_mapping.parquet"
    )

    # Summary (exclude extdc_waltman from standard summary if node set differs)
    summary_results = {k: v for k, v in all_results.items()}
    print_summary(summary_results, field_id, len(node_ids))
    log.info("Field %d done in %.1fs", field_id, time.time() - t_start)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Build DC/BC/CC edge lists")
    parser.add_argument("--field", default="34",
                        help="Field ID or 'all' for all sample fields")
    parser.add_argument("--link-type", nargs="*", default=None,
                        help="Link types to build (dc/bc/cc/extdc). Default: all")
    parser.add_argument("--min-shared", type=int, default=2,
                        help="Min shared refs/citers for BC/CC (default: 2)")
    args = parser.parse_args()

    _update_min_shared(args.min_shared)

    fields = SAMPLE_FIELDS if args.field == "all" else [int(args.field)]

    for fid in fields:
        build_field(fid, args.link_type)


if __name__ == "__main__":
    main()
