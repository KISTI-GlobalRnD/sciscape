#!/usr/bin/env python3
"""Test the CPM-density dendrogram pipeline on KRISS network data.

Usage:
    python scripts/test_dendrogram_pipeline.py [--n-nodes 5000] [--min-size 50]

Phases:
    1. Load edge table → build igraph → extract GCC
    2. (Optional) BFS subsample to --n-nodes
    3. Build greedy CPM-density dendrogram (Rust)
    4. Run constrained_cut with --min-size
    5. Print summary statistics
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
KRISS_DIR = Path.home() / "Desktop/Workspace/1.4.2.KRISS"
EDGE_PATH = KRISS_DIR / "Data" / "KRISS_pair_links" / "dc_bc_cc_total_pair.txt"


def load_graph(edge_path: Path, n_target: int | None = None, seed: int = 42):
    """Load edge table → igraph → GCC → optional BFS subsample."""
    import igraph as ig
    import polars as pl

    logger.info("Loading edge table: %s", edge_path)
    t0 = time.perf_counter()
    df = pl.read_csv(edge_path, separator="\t")
    logger.info("  %d edges loaded in %.1fs", len(df), time.perf_counter() - t0)

    # Build graph
    t0 = time.perf_counter()
    uid_col1, uid_col2, weight_col = df.columns[0], df.columns[1], df.columns[2]
    all_uids = list(set(df[uid_col1].to_list() + df[uid_col2].to_list()))
    uid_to_idx = {u: i for i, u in enumerate(all_uids)}

    sources = [uid_to_idx[u] for u in df[uid_col1].to_list()]
    targets = [uid_to_idx[u] for u in df[uid_col2].to_list()]
    weights = df[weight_col].to_list()

    g = ig.Graph(n=len(all_uids), edges=list(zip(sources, targets)), directed=False)
    g.es["weight"] = weights
    logger.info("  Graph: %d nodes, %d edges in %.1fs",
                g.vcount(), g.ecount(), time.perf_counter() - t0)

    # GCC
    t0 = time.perf_counter()
    gcc_ids = g.connected_components().giant().vs.indices
    g_gcc = g.subgraph(gcc_ids)
    logger.info("  GCC: %d nodes, %d edges in %.1fs",
                g_gcc.vcount(), g_gcc.ecount(), time.perf_counter() - t0)

    # Optional subsample
    if n_target is not None and g_gcc.vcount() > n_target:
        t0 = time.perf_counter()
        import random
        random.seed(seed)
        start = random.randint(0, g_gcc.vcount() - 1)
        bfs_order = g_gcc.bfs(start)[0]
        keep = bfs_order[:n_target]
        g_sub = g_gcc.subgraph(keep)
        # Re-extract GCC of subsample (BFS subsample should be connected)
        gcc2 = g_sub.connected_components().giant().vs.indices
        g_sub = g_sub.subgraph(gcc2)
        logger.info("  Subsample: %d nodes, %d edges in %.1fs",
                    g_sub.vcount(), g_sub.ecount(), time.perf_counter() - t0)
        return g_sub
    return g_gcc


def main():
    parser = argparse.ArgumentParser(description="Test dendrogram pipeline on KRISS data")
    parser.add_argument("--n-nodes", type=int, default=5000,
                        help="Target node count (0 = full GCC)")
    parser.add_argument("--min-size", type=int, default=50,
                        help="Minimum cluster size for constrained cut")
    parser.add_argument("--mode", type=str, default="cpm",
                        choices=["cpm", "triadic_cpm"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert EDGE_PATH.exists(), f"Edge file not found: {EDGE_PATH}"

    n_target = args.n_nodes if args.n_nodes > 0 else None
    graph = load_graph(EDGE_PATH, n_target=n_target, seed=args.seed)

    # --- Build dendrogram ---
    from sciscape.clustering.dendrogram import build_dendrogram
    from sciscape.clustering.constrained_cut import constrained_cut

    logger.info("Building dendrogram (mode=%s)...", args.mode)
    t0 = time.perf_counter()
    linkage = build_dendrogram(graph, mode=args.mode)
    dt_dendro = time.perf_counter() - t0
    logger.info("  Dendrogram: %d merges in %.2fs", len(linkage), dt_dendro)

    # Stats
    heights = linkage[:, 2]
    logger.info("  Height range: [%.6f, %.6f]", heights.min(), heights.max())
    logger.info("  Height median: %.6f", np.median(heights))
    n_zero = np.sum(heights == 0.0)
    logger.info("  Zero-height merges: %d (%.1f%%)", n_zero, 100 * n_zero / len(heights))

    # --- Constrained cut ---
    logger.info("Running constrained_cut (min_size=%d)...", args.min_size)
    t0 = time.perf_counter()
    result = constrained_cut(linkage, min_size=args.min_size)
    dt_cut = time.perf_counter() - t0
    logger.info("  Cut: %d clusters in %.2fs (feasible=%s)",
                result.n_clusters, dt_cut, result.feasible)
    logger.info("  Total stability: %.6f", result.total_stability)

    # Cluster size distribution
    sizes = sorted([len(c) for c in result.partition], reverse=True)
    logger.info("  Cluster sizes (top 10): %s", sizes[:10])
    logger.info("  Min: %d, Max: %d, Median: %d",
                min(sizes), max(sizes), int(np.median(sizes)))

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Graph: %d nodes, %d edges", graph.vcount(), graph.ecount())
    logger.info("  Mode: %s", args.mode)
    logger.info("  Dendrogram: %.2fs", dt_dendro)
    logger.info("  Constrained cut (k=%d): %d clusters in %.2fs",
                args.min_size, result.n_clusters, dt_cut)
    logger.info("  Feasible: %s", result.feasible)


if __name__ == "__main__":
    main()
