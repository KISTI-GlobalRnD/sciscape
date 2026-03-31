#!/usr/bin/env python3
"""Combine link types (DC+BC, DC+CC, DC+BC+CC) with various combination methods.

Methods:
- sum:      w_combined = w_A + w_B  (after per-type normalization to [0,1])
- max:      w_combined = max(w_A, w_B)
- noisy_or: w_combined = 1 - (1-w_A)(1-w_B)  (requires weights in [0,1])

Before combining, each link type is normalized to [0,1] by dividing by its max weight.

Usage:
    python scripts/combine_linktype_edges.py --field 34
    python scripts/combine_linktype_edges.py --field 34 --method sum --types dc_fractional bc_cosine cc_cosine
"""
from __future__ import annotations

import argparse
import logging
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("combine")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges"

# Default best normalizations per link type (from literature)
BEST_NORM = {
    "dc": "dc_fractional",
    "bc": "bc_cosine",
    "cc": "cc_cosine",
    "extdc": "extdc_fractional",
}


def load_sparse(field_id: int, link_type: str, node_ids: list[str], id2idx: dict[str, int]):
    """Load edge list and return as sparse matrix (n×n)."""
    path = EDGE_DIR / f"field_{field_id}" / f"{link_type}.parquet"
    df = pl.read_parquet(path)
    n = len(node_ids)

    rows, cols, weights = [], [], []
    for src, dst, w in zip(df["src"].to_list(), df["dst"].to_list(), df["weight"].to_list()):
        if src in id2idx and dst in id2idx:
            i, j = id2idx[src], id2idx[dst]
            rows.append(i)
            cols.append(j)
            weights.append(w)
            rows.append(j)
            cols.append(i)
            weights.append(w)

    M = sparse.csr_matrix(
        (np.array(weights), (np.array(rows), np.array(cols))),
        shape=(n, n),
    )
    M = M.maximum(M.T)  # ensure symmetric
    log.info("  Loaded %s: %d edges, weight range [%.6f, %.6f]",
             link_type, M.nnz // 2, M.data.min(), M.data.max())
    return M


def normalize_01(M: sparse.csr_matrix) -> sparse.csr_matrix:
    """Normalize weights to [0, 1] by dividing by max."""
    mx = M.data.max()
    if mx > 0:
        M = M.copy()
        M.data = M.data / mx
    return M


def combine_sum(matrices: list[sparse.csr_matrix]) -> sparse.csr_matrix:
    """Sum of normalized weights."""
    result = matrices[0].copy()
    for M in matrices[1:]:
        result = result + M
    return result


def combine_max(matrices: list[sparse.csr_matrix]) -> sparse.csr_matrix:
    """Element-wise maximum."""
    result = matrices[0].copy()
    for M in matrices[1:]:
        result = result.maximum(M)
    return result


def combine_noisy_or(matrices: list[sparse.csr_matrix]) -> sparse.csr_matrix:
    """Noisy-OR: 1 - prod(1 - w_i). Weights must be in [0, 1]."""
    n = matrices[0].shape[0]

    # Collect all edges across all matrices
    all_edges: dict[tuple[int, int], list[float]] = {}
    for M in matrices:
        coo = M.tocoo()
        for i, j, w in zip(coo.row, coo.col, coo.data):
            if i < j:
                key = (i, j)
                if key not in all_edges:
                    all_edges[key] = []
                all_edges[key].append(w)

    rows, cols, weights = [], [], []
    for (i, j), ws in all_edges.items():
        # noisy-OR
        prob = 1.0 - np.prod([1.0 - w for w in ws])
        rows.extend([i, j])
        cols.extend([j, i])
        weights.extend([prob, prob])

    return sparse.csr_matrix(
        (np.array(weights), (np.array(rows), np.array(cols))),
        shape=(n, n),
    )


def sparse_to_edgelist(M: sparse.csr_matrix, idx2id: list[str]) -> pl.DataFrame:
    """Convert upper triangle of sparse matrix to edge list DataFrame."""
    upper = sparse.triu(M, k=1).tocoo()
    return pl.DataFrame({
        "src": [idx2id[i] for i in upper.row],
        "dst": [idx2id[i] for i in upper.col],
        "weight": upper.data.astype(np.float64),
    })


def build_combinations(field_id: int, link_types: list[str], methods: list[str]):
    """Build all requested combinations."""
    # Load node mapping
    mapping = pl.read_parquet(EDGE_DIR / f"field_{field_id}" / "node_mapping.parquet")
    node_ids = mapping["work_id"].to_list()
    id2idx = {wid: i for i, wid in enumerate(node_ids)}
    n = len(node_ids)

    # Load and normalize each link type
    matrices = {}
    for lt in link_types:
        M = load_sparse(field_id, lt, node_ids, id2idx)
        M_norm = normalize_01(M)
        matrices[lt] = M_norm
        log.info("    %s normalized: nnz=%d", lt, M_norm.nnz // 2)

    # Generate all 2-way and 3-way combinations
    type_names = list(matrices.keys())
    combos = []
    for r in range(2, len(type_names) + 1):
        for combo in combinations(type_names, r):
            combos.append(combo)

    log.info("Generating %d combinations × %d methods = %d edge sets",
             len(combos), len(methods), len(combos) * len(methods))

    out_dir = EDGE_DIR / f"field_{field_id}"
    results = []

    for combo in combos:
        combo_matrices = [matrices[lt] for lt in combo]
        combo_label = "+".join(lt.split("_")[0] for lt in combo)  # e.g., "dc+bc"

        for method in methods:
            if method == "sum":
                C = combine_sum(combo_matrices)
            elif method == "max":
                C = combine_max(combo_matrices)
            elif method == "noisy_or":
                C = combine_noisy_or(combo_matrices)
            else:
                raise ValueError(f"Unknown method: {method}")

            name = f"combo_{combo_label}_{method}"
            n_edges = sparse.triu(C, k=1).nnz
            w_upper = sparse.triu(C, k=1).data

            log.info("  %s: %d edges, weight [%.6f, %.6f]",
                     name, n_edges, w_upper.min(), w_upper.max())

            # Save
            df = sparse_to_edgelist(C, node_ids)
            path = out_dir / f"{name}.parquet"
            df.write_parquet(path)
            results.append((name, n_edges, float(w_upper.mean()), float(w_upper.max())))

    # Summary
    print(f"\n{'='*80}")
    print(f"COMBINATION RESULTS — Field {field_id}")
    print(f"{'='*80}")
    print(f"{'Name':<35} {'Edges':>10} {'Mean Weight':>12} {'Max':>8}")
    print("-" * 68)
    for name, n_e, mean_w, max_w in results:
        print(f"{name:<35} {n_e:>10,} {mean_w:>12.6f} {max_w:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Combine link type edge lists")
    parser.add_argument("--field", type=int, default=34)
    parser.add_argument("--types", nargs="*", default=None,
                        help="Link types to combine (default: dc_fractional bc_cosine cc_cosine)")
    parser.add_argument("--method", nargs="*", default=["sum", "max", "noisy_or"],
                        help="Combination methods")
    args = parser.parse_args()

    types = args.types or [BEST_NORM["dc"], BEST_NORM["bc"], BEST_NORM["cc"]]
    build_combinations(args.field, types, args.method)


if __name__ == "__main__":
    main()
