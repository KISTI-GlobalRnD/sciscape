#!/usr/bin/env python3
"""Sample nodes and inspect their top/bottom neighbors per link type.

For each sampled node, show:
  - Target node: work_id, subfield, year, cited_by_count
  - Top-5 / Bottom-5 neighbors per link type (DC, BC, CC)
  - Neighbor subfields, weights, overlap across types

Usage:
    .venv/bin/python scripts/inspect_neighbors.py --field 15 --n-sample 10
    .venv/bin/python scripts/inspect_neighbors.py --field 15 --n-sample 5 --fetch-titles
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inspect")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
HDD = Path.home() / "Desktop/HDD/local_map_analysis_data/processed/outputs/data"
META_DIR = HDD / "oa26_gcc_only"

LINK_TYPES = {
    "DC": "dc_fractional",
    "BC": "bc_assoc_strength",
    "CC": "cc_assoc_strength",
    "Emb_full": "emb_full_knn30",
    "Emb_bg": "emb_bg_knn30",
    "Emb_nov": "emb_nov_knn30",
}


def load_meta(field_id: int) -> pl.DataFrame:
    """Load node metadata (subfield, year, citations)."""
    path = META_DIR / f"field_{field_id}_nodes_oa_meta_gcc.parquet"
    df = pl.read_parquet(path)
    return df


def load_edges(field_id: int, link_type: str) -> pl.DataFrame:
    """Load edge list."""
    path = EDGE_DIR / f"field_{field_id}" / f"{link_type}.parquet"
    return pl.read_parquet(path)


def get_neighbors(
    edges: pl.DataFrame,
    node_id: str,
    top_k: int = 5,
    bottom_k: int = 5,
    topn: int = 30,
) -> tuple[pl.DataFrame, pl.DataFrame, int, int]:
    """Get top-k strongest and bottom-k weakest neighbors for a node.

    First applies top-N filtering (default 30), then returns the
    top-k and bottom-k within that filtered set. This simulates
    the actual pipeline behavior where each node keeps only its
    top-N strongest neighbors.

    Returns (top_df, bottom_df, total_raw, total_after_topn).
    """
    # Edges where node is uid1 or uid2
    nbrs = edges.filter(
        (pl.col("uid1") == node_id) | (pl.col("uid2") == node_id)
    )
    # Normalize: always put target in 'src', neighbor in 'nbr'
    nbrs = nbrs.with_columns(
        pl.when(pl.col("uid1") == node_id)
        .then(pl.col("uid2"))
        .otherwise(pl.col("uid1"))
        .alias("nbr_id"),
    ).select(["nbr_id", "rel_sum2"])

    # Deduplicate (same neighbor may appear in both directions)
    nbrs = nbrs.group_by("nbr_id").agg(pl.col("rel_sum2").max())

    total_raw = nbrs.height

    # Apply top-N filter (simulate backbone)
    nbrs = nbrs.sort("rel_sum2", descending=True).head(topn)
    total_topn = nbrs.height

    top = nbrs.head(top_k)
    bottom = nbrs.tail(bottom_k)

    return top, bottom, total_raw, total_topn


def fetch_title_from_openalex(work_id: str) -> str:
    """Fetch title from OpenAlex API."""
    import urllib.request
    oa_id = work_id.replace("W", "")
    url = f"https://api.openalex.org/works/W{oa_id}?select=title"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("title", "N/A")
    except Exception:
        return "N/A"


def inspect_node(
    node_id: str,
    meta: pl.DataFrame,
    edges_dict: dict[str, pl.DataFrame],
    fetch_titles: bool = False,
    top_k: int = 5,
    bottom_k: int = 5,
    topn: int = 30,
) -> dict:
    """Inspect one node's neighborhoods across link types."""
    # Target node info
    node_meta = meta.filter(pl.col("work_id") == node_id)
    if node_meta.height == 0:
        return None

    row = node_meta.row(0, named=True)
    result = {
        "work_id": node_id,
        "subfield": row.get("subfield_name", "N/A"),
        "year": row.get("publication_year", "N/A"),
        "cited_by": row.get("cited_by_count", 0),
    }

    if fetch_titles:
        result["title"] = fetch_title_from_openalex(node_id)

    # Build id→meta lookup
    meta_dict = {}
    for r in meta.select(["work_id", "subfield_name"]).iter_rows():
        meta_dict[r[0]] = r[1]

    result["neighbors"] = {}
    for label, edges in edges_dict.items():
        top, bottom, total_raw, total_topn = get_neighbors(
            edges, node_id, top_k, bottom_k, topn=topn,
        )

        def annotate(df: pl.DataFrame) -> list[dict]:
            rows = []
            for r in df.iter_rows(named=True):
                nbr_id = r["nbr_id"]
                entry = {
                    "work_id": nbr_id,
                    "weight": round(float(r["rel_sum2"]), 6),
                    "subfield": meta_dict.get(nbr_id, "?"),
                }
                if fetch_titles:
                    entry["title"] = fetch_title_from_openalex(nbr_id)
                rows.append(entry)
            return rows

        result["neighbors"][label] = {
            "total_raw": total_raw,
            "total_topn": total_topn,
            "top": annotate(top),
            "bottom": annotate(bottom),
        }

    return result


def print_node_report(result: dict):
    """Pretty-print one node's report (title-focused)."""
    print(f"\n{'='*100}")
    title_str = result.get('title', 'N/A')
    print(f"  TARGET: {result['work_id']} | year={result['year']} | cited_by={result['cited_by']}")
    print(f"  TITLE: {title_str}")
    print(f"{'='*100}")

    for label, nbr_info in result["neighbors"].items():
        print(f"\n  [{label}] raw={nbr_info['total_raw']} → top-{nbr_info['total_topn']} used")

        print(f"  {'─'*90}")
        print(f"  TOP-{len(nbr_info['top'])} (strongest)")
        for i, n in enumerate(nbr_info["top"], 1):
            title_str = n.get('title', n['work_id'])
            print(f"    {i}. w={n['weight']:.6f} | {title_str}")

        print(f"  BOTTOM-{len(nbr_info['bottom'])} (weakest)")
        for i, n in enumerate(nbr_info["bottom"], 1):
            title_str = n.get('title', n['work_id'])
            print(f"    {i}. w={n['weight']:.6f} | {title_str}")


def compute_overlap_stats(results: list[dict]) -> None:
    """Compute and print pairwise neighbor overlap between link types."""
    labels = list(LINK_TYPES.keys())

    print(f"\n{'='*100}")
    print(f"  NEIGHBOR OVERLAP — Top-k sets across link types ({len(results)} nodes)")
    print(f"{'='*100}")

    for r in results:
        top_sets = {}
        for label in labels:
            nbrs = r["neighbors"].get(label, {})
            top_sets[label] = {n["work_id"] for n in nbrs.get("top", [])}

        overlap_str = []
        for i, a in enumerate(labels):
            for b in labels[i+1:]:
                inter = len(top_sets[a] & top_sets[b])
                union = len(top_sets[a] | top_sets[b])
                jaccard = inter / union if union > 0 else 0
                overlap_str.append(f"{a}∩{b}={inter}")
        print(f"  {r['work_id']}: {', '.join(overlap_str)}")

    # Aggregate
    print(f"\n  Average top-k Jaccard overlap:")
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            jaccards = []
            for r in results:
                sa = {n["work_id"] for n in r["neighbors"].get(a, {}).get("top", [])}
                sb = {n["work_id"] for n in r["neighbors"].get(b, {}).get("top", [])}
                union = len(sa | sb)
                jaccards.append(len(sa & sb) / union if union > 0 else 0)
            import numpy as np
            print(f"    {a} ∩ {b}: J={np.mean(jaccards):.3f} (±{np.std(jaccards):.3f})")


def main():
    parser = argparse.ArgumentParser(description="Inspect node neighborhoods per link type")
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--n-sample", type=int, default=10,
                        help="Number of nodes to sample")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bottom-k", type=int, default=5)
    parser.add_argument("--topn", type=int, default=30,
                        help="Top-N backbone filter per node (default: 30)")
    parser.add_argument("--fetch-titles", action="store_true",
                        help="Fetch titles from OpenAlex API (slow)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degree-tier", choices=["high", "mid", "low", "all"], default="all",
                        help="Sample from specific degree tier")
    args = parser.parse_args()

    field_id = args.field

    # Load metadata
    log.info("Loading metadata...")
    meta = load_meta(field_id)
    log.info("  %d nodes in metadata", meta.height)

    # Load all edge types
    log.info("Loading edge data...")
    edges_dict = {}
    for label, filename in LINK_TYPES.items():
        path = EDGE_DIR / f"field_{field_id}" / f"{filename}.parquet"
        if not path.exists():
            log.warning("  %s: file not found, skipping (%s)", label, path)
            continue
        edges = load_edges(field_id, filename)
        edges_dict[label] = edges
        log.info("  %s: %d edges", label, edges.height)

    # Find common nodes (present in ALL link types)
    node_sets = []
    for label, edges in edges_dict.items():
        ids = set(edges["uid1"].to_list()) | set(edges["uid2"].to_list())
        node_sets.append(ids)
        log.info("  %s covers %d nodes", label, len(ids))

    common = set.intersection(*node_sets)
    log.info("Common nodes (in all %d types): %d", len(edges_dict), len(common))

    # Compute degree in BC (most edges) for stratified sampling
    bc_edges = edges_dict["BC"]
    bc_degree = (
        pl.concat([
            bc_edges.select(pl.col("uid1").alias("id")),
            bc_edges.select(pl.col("uid2").alias("id")),
        ])
        .filter(pl.col("id").is_in(pl.Series(sorted(common))))
        .group_by("id")
        .len()
        .rename({"len": "degree"})
    )

    # Stratified sampling by degree
    random.seed(args.seed)
    candidates = bc_degree.sort("degree")
    n = candidates.height

    if args.degree_tier == "high":
        pool = candidates.tail(n // 5)["id"].to_list()
    elif args.degree_tier == "mid":
        pool = candidates.slice(n * 2 // 5, n // 5)["id"].to_list()
    elif args.degree_tier == "low":
        pool = candidates.head(n // 5)["id"].to_list()
    else:
        # Stratified: mix of high/mid/low
        high = candidates.tail(n // 5)["id"].to_list()
        mid = candidates.slice(n * 2 // 5, n // 5)["id"].to_list()
        low = candidates.head(n // 5)["id"].to_list()
        n_each = max(1, args.n_sample // 3)
        pool = (
            random.sample(high, min(n_each, len(high)))
            + random.sample(mid, min(n_each, len(mid)))
            + random.sample(low, min(args.n_sample - 2 * n_each, len(low)))
        )

    if args.degree_tier != "all":
        sample_ids = random.sample(pool, min(args.n_sample, len(pool)))
    else:
        sample_ids = pool[:args.n_sample]

    log.info("Sampled %d nodes", len(sample_ids))

    # Use full meta for lookup (not filtered to common)
    meta_subset = meta

    # Inspect each node
    results = []
    for i, node_id in enumerate(sample_ids):
        log.info("Inspecting node %d/%d: %s", i + 1, len(sample_ids), node_id)
        result = inspect_node(
            node_id, meta_subset, edges_dict,
            fetch_titles=args.fetch_titles,
            top_k=args.top_k, bottom_k=args.bottom_k,
            topn=args.topn,
        )
        if result:
            results.append(result)
            print_node_report(result)

    # Summary — neighbor overlap across link types
    compute_overlap_stats(results)


if __name__ == "__main__":
    main()
