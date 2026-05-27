"""Prepare cluster network data for D3.js visualization.

Builds a JSON structure with:
- nodes: clusters with size, label, hierarchy level
- edges: inter-cluster connections per layer (DC, BC, CC, combined)
- hierarchy: parent-child relationships across levels
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def _keyword_label_column(cols: list[str]) -> str:
    for preferred in ("display_label", "keyword", "term", "word"):
        for col in cols:
            if col == preferred or preferred in col.lower():
                return col
    return cols[1] if len(cols) > 1 else cols[0]


def _keyword_score_column(cols: list[str]) -> str | None:
    for preferred in ("quality_score", "score", "weight", "tfidf"):
        for col in cols:
            if col == preferred or preferred in col.lower():
                return col
    return None


def build_network_json(
    edges_path: Path,
    membership_path: Path | None = None,
    keywords_path: Path | None = None,
    *,
    edge_layer_paths: Dict[str, Path] | None = None,
    max_edges: int = 2000,
) -> Dict[str, Any]:
    """Build cluster network JSON for D3 visualization.

    Parameters
    ----------
    edges_path : Path
        Combined edge table (uid1, uid2, rel_sum2).
    membership_path : Path, optional
        Membership table (uid, cluster_nano, cluster_micro, ...).
    keywords_path : Path, optional
        Keywords table for cluster labels.
    edge_layer_paths : dict, optional
        Per-layer edge files: {"dc": path, "bc": path, "cc": path}.
    max_edges : int
        Maximum edges to include (top by weight).

    Returns
    -------
    dict
        JSON-serializable structure for D3.
    """
    result: Dict[str, Any] = {"nodes": [], "edges": [], "layers": [], "levels": []}

    # Load combined edges
    edges = pl.read_parquet(edges_path)

    # Load membership
    levels: List[str] = []
    mem_df = None
    if membership_path and Path(membership_path).exists():
        mem_df = pl.read_parquet(membership_path)
        levels = [c.replace("cluster_", "") for c in mem_df.columns if c.startswith("cluster_")]
        result["levels"] = levels

    if not levels:
        levels = ["default"]

    # Build network per level
    for level in levels:
        cluster_col = f"cluster_{level}" if mem_df is not None else None
        net = _build_level_network(
            edges, mem_df, cluster_col, level,
            edge_layer_paths=edge_layer_paths,
            keywords_path=keywords_path,
            max_edges=max_edges,
        )
        result["nodes"].append({"level": level, "data": net["nodes"]})
        result["edges"].append({"level": level, "data": net["edges"]})

    # Available layers
    layer_names = ["combined"]
    if edge_layer_paths:
        layer_names.extend(sorted(edge_layer_paths.keys()))
    result["layers"] = layer_names

    # Hierarchy links (parent-child between levels)
    if mem_df is not None and len(levels) >= 2:
        result["hierarchy"] = _build_hierarchy_links(mem_df, levels)

    return result


def _build_level_network(
    edges: pl.DataFrame,
    mem_df: pl.DataFrame | None,
    cluster_col: str | None,
    level: str,
    *,
    edge_layer_paths: Dict[str, Path] | None = None,
    keywords_path: Path | None = None,
    max_edges: int = 2000,
) -> Dict[str, Any]:
    """Build nodes + edges for one hierarchy level."""

    # Determine cluster assignments
    if mem_df is not None and cluster_col and cluster_col in mem_df.columns:
        uid_to_cluster = dict(zip(
            mem_df["uid"].to_list(),
            mem_df[cluster_col].to_list(),
        ))
    else:
        # No membership: each uid is its own node
        all_uids_series = pl.concat([edges["uid1"], edges["uid2"]]).unique().sort()
        uid_to_cluster = {uid: i for i, uid in enumerate(all_uids_series.to_list())}

    # Cluster sizes
    cluster_sizes = Counter(uid_to_cluster.values())
    n_total = sum(cluster_sizes.values())

    # Cluster keywords (labels)
    cluster_labels: Dict[int, str] = {}
    if keywords_path and Path(keywords_path).exists():
        try:
            kw_df = pl.read_parquet(keywords_path)
            # Expect columns: cluster, keyword, score (or similar)
            for col in kw_df.columns:
                if "cluster" in col.lower():
                    cluster_key = col
                    break
            else:
                cluster_key = kw_df.columns[0]
            keyword_key = _keyword_label_column(kw_df.columns)
            score_key = _keyword_score_column(kw_df.columns)

            # Top keyword per cluster (vectorized: first keyword per group)
            if score_key:
                first_kw = (
                    kw_df.sort(score_key, descending=True)
                    .group_by(cluster_key, maintain_order=True)
                    .agg(pl.col(keyword_key).first())
                )
            else:
                first_kw = kw_df.group_by(cluster_key).agg(pl.col(keyword_key).first())
            cluster_labels = dict(zip(
                first_kw[cluster_key].cast(pl.Int64).to_list(),
                first_kw[keyword_key].cast(pl.Utf8).to_list(),
            ))
        except Exception:
            pass

    # Compute per-cluster year stats from abstracts if available
    cluster_year_stats: Dict[int, Dict] = {}
    if mem_df is not None and cluster_col and "pubyear" in mem_df.columns:
        year_agg = (
            mem_df.filter(pl.col("pubyear").is_not_null() & (pl.col("pubyear") > 0))
            .with_columns(pl.col("uid").replace_strict(uid_to_cluster, default=None).alias("_cl"))
            .filter(pl.col("_cl").is_not_null())
            .group_by("_cl")
            .agg(
                pl.col("pubyear").mean().alias("avg_year"),
                pl.col("pubyear").min().alias("min_year"),
                pl.col("pubyear").max().alias("max_year"),
            )
        )
        for row in year_agg.iter_rows():
            cluster_year_stats[int(row[0])] = {
                "avg_year": round(row[1], 1) if row[1] else 0,
                "year_range": [int(row[2]) if row[2] else 0, int(row[3]) if row[3] else 0],
            }

    # Build nodes
    nodes = []
    for cid in sorted(cluster_sizes.keys()):
        size = cluster_sizes[cid]
        ys = cluster_year_stats.get(int(cid), {})
        nodes.append({
            "id": int(cid),
            "size": size,
            "pct": round(100 * size / n_total, 2) if n_total else 0,
            "label": cluster_labels.get(int(cid), f"C{cid}"),
            "avg_year": ys.get("avg_year", 0),
            "year_range": ys.get("year_range", [0, 0]),
            "year": ys.get("avg_year", 0),
            "citations": 0,
        })

    # Build edges (combined + per-layer)
    all_edges: Dict[str, List[Dict]] = {"combined": []}

    # Combined
    combined_cluster_edges = _aggregate_to_cluster_edges(edges, uid_to_cluster, max_edges)
    all_edges["combined"] = combined_cluster_edges

    # Per-layer
    if edge_layer_paths:
        for layer_name, layer_path in edge_layer_paths.items():
            if Path(layer_path).exists():
                try:
                    layer_df = pl.read_parquet(layer_path)
                    layer_edges = _aggregate_to_cluster_edges(layer_df, uid_to_cluster, max_edges)
                    all_edges[layer_name] = layer_edges
                except Exception:
                    pass

    return {"nodes": nodes, "edges": all_edges}


def _aggregate_to_cluster_edges(
    edges: pl.DataFrame,
    uid_to_cluster: Dict,
    max_edges: int,
) -> List[Dict]:
    """Aggregate paper-level edges to cluster-level, sum weights."""
    # Vectorized: join cluster IDs, filter, aggregate
    mapping = pl.DataFrame({
        "uid": list(uid_to_cluster.keys()),
        "_cl": list(uid_to_cluster.values()),
    })
    agg = (
        edges
        .join(mapping.rename({"uid": "uid1", "_cl": "_c1"}), on="uid1", how="inner")
        .join(mapping.rename({"uid": "uid2", "_cl": "_c2"}), on="uid2", how="inner")
        .filter(pl.col("_c1") != pl.col("_c2"))
        .with_columns(
            pl.min_horizontal("_c1", "_c2").alias("_lo"),
            pl.max_horizontal("_c1", "_c2").alias("_hi"),
        )
        .group_by(["_lo", "_hi"])
        .agg(pl.col("rel_sum2").sum().alias("_w"))
        .sort("_w", descending=True)
        .head(max_edges)
    )

    if agg.height == 0:
        return []

    sorted_edges = list(zip(
        zip(agg["_lo"].to_list(), agg["_hi"].to_list()),
        agg["_w"].to_list(),
    ))

    if not sorted_edges:
        return []

    max_w = sorted_edges[0][1] if sorted_edges else 1.0
    return [
        {
            "source": int(pair[0]),
            "target": int(pair[1]),
            "weight": round(w, 4),
            "norm": round(w / max_w, 4),
        }
        for pair, w in sorted_edges
    ]


def _build_hierarchy_links(
    mem_df: pl.DataFrame,
    levels: List[str],
) -> List[Dict]:
    """Build parent-child links between adjacent hierarchy levels."""
    links = []
    for i in range(len(levels) - 1):
        child_col = f"cluster_{levels[i]}"
        parent_col = f"cluster_{levels[i + 1]}"
        if child_col not in mem_df.columns or parent_col not in mem_df.columns:
            continue

        # Find dominant parent for each child cluster
        pairs = mem_df.select(child_col, parent_col).group_by(child_col).agg(
            pl.col(parent_col).mode().first().alias("parent")
        )
        child_level = levels[i]
        parent_level = levels[i + 1]
        for child_id, parent_id in zip(pairs[child_col].to_list(), pairs["parent"].to_list()):
            links.append({
                "child_level": child_level,
                "parent_level": parent_level,
                "child": int(child_id),
                "parent": int(parent_id),
            })

    return links


def build_term_network_json(
    keywords_path: Path,
    *,
    top_k_per_cluster: int = 10,
    min_cooc: int = 2,
    max_terms: int = 200,
    max_edges: int = 1500,
) -> Dict[str, Any]:
    """Build term co-occurrence network JSON for D3 visualization.

    Nodes = keywords (sized by global frequency, colored by dominant cluster).
    Edges = co-occurrence within same cluster (weight = number of shared clusters).

    Parameters
    ----------
    keywords_path : Path
        Keywords parquet (cluster, keyword, score columns).
    top_k_per_cluster : int
        Top keywords per cluster to include.
    min_cooc : int
        Minimum co-occurrence count for an edge.
    max_terms : int
        Maximum terms to show.
    max_edges : int
        Maximum edges to show.
    """
    kw_df = pl.read_parquet(keywords_path)
    cols = kw_df.columns

    # Auto-detect columns
    cluster_col = next((c for c in cols if "cluster" in c.lower()), cols[0])
    keyword_col = _keyword_label_column(cols)
    score_col = _keyword_score_column(cols)

    # Collect top-k keywords per cluster
    term_clusters: Dict[str, List[int]] = defaultdict(list)  # term → [cluster_ids]
    term_scores: Dict[str, float] = defaultdict(float)       # term → max score
    term_dominant: Dict[str, int] = {}                        # term → dominant cluster

    # Top-k keywords per cluster (vectorized: sort + group_by head)
    if score_col and score_col in kw_df.columns:
        top_all = (
            kw_df.sort(score_col, descending=True)
            .group_by(cluster_col, maintain_order=True)
            .head(top_k_per_cluster)
        )
    else:
        top_all = kw_df.group_by(cluster_col, maintain_order=True).head(top_k_per_cluster)

    for term, cid, score in zip(
        top_all[keyword_col].cast(pl.Utf8).to_list(),
        top_all[cluster_col].to_list(),
        top_all[score_col].to_list() if score_col and score_col in top_all.columns else [1.0] * top_all.height,
    ):
        term = str(term)
        cid = int(cid)
        score = float(score)
        term_clusters[term].append(cid)
        if score > term_scores[term]:
            term_scores[term] = score
            term_dominant[term] = cid

    # Select top terms by score
    sorted_terms = sorted(term_scores.keys(), key=lambda t: -term_scores[t])[:max_terms]
    term_to_idx = {t: i for i, t in enumerate(sorted_terms)}
    cluster_ids = sorted({cid for term in sorted_terms for cid in term_clusters.get(term, [])})

    # Build nodes
    nodes = []
    for t in sorted_terms:
        nodes.append({
            "id": term_to_idx[t],
            "label": t,
            "score": round(term_scores[t], 4),
            "cluster": term_dominant.get(t, 0),
            "n_clusters": len(set(term_clusters[t])),
            "frequency": len(term_clusters[t]),
        })

    # Build edges: terms that co-occur in the same cluster
    edge_counts: Dict[tuple, int] = defaultdict(int)
    for cid in cluster_ids:
        cluster_terms = [t for t in sorted_terms if cid in term_clusters.get(t, [])]
        for i in range(len(cluster_terms)):
            for j in range(i + 1, len(cluster_terms)):
                a, b = cluster_terms[i], cluster_terms[j]
                pair = (min(term_to_idx[a], term_to_idx[b]),
                        max(term_to_idx[a], term_to_idx[b]))
                edge_counts[pair] += 1

    # Filter and sort edges
    edges = []
    max_w = max(edge_counts.values()) if edge_counts else 1
    for (s, t), count in sorted(edge_counts.items(), key=lambda x: -x[1])[:max_edges]:
        if count < min_cooc:
            continue
        edges.append({
            "source": s,
            "target": t,
            "weight": count,
            "norm": round(count / max_w, 4),
        })

    return {"nodes": nodes, "edges": edges, "n_clusters": len(cluster_ids)}


def build_temporal_snapshots(
    edges_path: Path,
    membership_path: Path,
    abstracts_path: Path | None = None,
    *,
    year_col: str = "pubyear",
) -> Dict[str, Any]:
    """Build per-year cluster network snapshots for temporal playback.

    Returns {year: {nodes: [...], edges: [...]}} for each year present.
    """
    edges = pl.read_parquet(edges_path)
    mem_df = pl.read_parquet(membership_path)
    cluster_col = next((c for c in mem_df.columns if c.startswith("cluster_")), None)
    if not cluster_col:
        return {"snapshots": {}, "years": []}

    # Need abstracts for year info
    if abstracts_path and Path(abstracts_path).exists():
        abs_df = pl.read_parquet(abstracts_path)
        if year_col in abs_df.columns and "uid" in abs_df.columns:
            mem_df = mem_df.join(abs_df.select("uid", year_col), on="uid", how="left")

    if year_col not in mem_df.columns:
        return {"snapshots": {}, "years": []}

    uid_to_cluster = dict(zip(mem_df["uid"].to_list(), mem_df[cluster_col].to_list()))
    uid_to_year = dict(zip(mem_df["uid"].to_list(), mem_df[year_col].to_list()))

    years = sorted(set(y for y in uid_to_year.values() if y and y > 0))
    if not years:
        return {"snapshots": {}, "years": []}

    # Pre-compute: node table with year + cluster for vectorized snapshots
    node_df = pl.DataFrame({
        "uid": list(uid_to_year.keys()),
        "_yr": [uid_to_year[u] or 0 for u in uid_to_year],
        "_cl": [uid_to_cluster.get(u) for u in uid_to_year],
    }).filter(pl.col("_cl").is_not_null())

    # Pre-compute: edges with cluster mapping
    cl_map = pl.DataFrame({"uid": list(uid_to_cluster.keys()), "_cl": list(uid_to_cluster.values())})
    edge_mapped = (
        edges
        .join(cl_map.rename({"uid": "uid1", "_cl": "_c1"}), on="uid1", how="inner")
        .join(cl_map.rename({"uid": "uid2", "_cl": "_c2"}), on="uid2", how="inner")
        .filter(pl.col("_c1") != pl.col("_c2"))
    )
    # Add year info for filtering
    yr_map = pl.DataFrame({"uid": list(uid_to_year.keys()), "_yr": [uid_to_year[u] or 0 for u in uid_to_year]})
    edge_mapped = (
        edge_mapped
        .join(yr_map.rename({"uid": "uid1", "_yr": "_yr1"}), on="uid1", how="left")
        .join(yr_map.rename({"uid": "uid2", "_yr": "_yr2"}), on="uid2", how="left")
        .with_columns(
            pl.max_horizontal("_yr1", "_yr2").alias("_max_yr"),
            pl.min_horizontal("_c1", "_c2").alias("_lo"),
            pl.max_horizontal("_c1", "_c2").alias("_hi"),
        )
    )

    # Cumulative snapshots: for each cutoff, filter by year
    snapshots = {}
    for cutoff in years:
        # Cluster sizes: count nodes with year <= cutoff
        cs = (
            node_df.filter(pl.col("_yr") <= cutoff)
            .group_by("_cl").len(name="_sz")
        )
        cluster_sizes = dict(zip(cs["_cl"].to_list(), cs["_sz"].to_list()))

        # Edge weights: filter edges where both endpoints ≤ cutoff
        ew = (
            edge_mapped.filter(pl.col("_max_yr") <= cutoff)
            .group_by(["_lo", "_hi"])
            .agg(pl.col("rel_sum2").sum().alias("_w"))
        )
        edge_weights = {
            (lo, hi): w
            for lo, hi, w in zip(ew["_lo"].to_list(), ew["_hi"].to_list(), ew["_w"].to_list())
        }

        n_total = sum(cluster_sizes.values()) or 1
        nodes = [
            {"id": int(cid), "size": sz, "pct": round(100 * sz / n_total, 1)}
            for cid, sz in sorted(cluster_sizes.items())
            if sz > 0
        ]
        top_edges = sorted(edge_weights.items(), key=lambda x: -x[1])[:500]
        max_w = top_edges[0][1] if top_edges else 1
        edge_list = [
            {"source": int(p[0]), "target": int(p[1]),
             "weight": round(w, 3), "norm": round(w / max_w, 3)}
            for p, w in top_edges
        ]
        snapshots[str(cutoff)] = {"nodes": nodes, "edges": edge_list}

    return {"snapshots": snapshots, "years": [str(y) for y in years]}


def find_bridge_papers(
    edges_path: Path,
    membership_path: Path,
    abstracts_path: Path | None,
    cluster_a: int,
    cluster_b: int,
    *,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Find papers that bridge two clusters (cross-cluster edges)."""
    edges = pl.read_parquet(edges_path)
    mem_df = pl.read_parquet(membership_path)
    cluster_col = next((c for c in mem_df.columns if c.startswith("cluster_")), None)
    if not cluster_col:
        return []

    uid_to_cluster = dict(zip(mem_df["uid"].to_list(), mem_df[cluster_col].to_list()))

    # Find edges between cluster_a and cluster_b (vectorized)
    cl_map = pl.DataFrame({"uid": list(uid_to_cluster.keys()), "_cl": list(uid_to_cluster.values())})
    bridge_edges = (
        edges
        .join(cl_map.rename({"uid": "uid1", "_cl": "_c1"}), on="uid1", how="inner")
        .join(cl_map.rename({"uid": "uid2", "_cl": "_c2"}), on="uid2", how="inner")
        .filter(
            ((pl.col("_c1") == cluster_a) & (pl.col("_c2") == cluster_b)) |
            ((pl.col("_c1") == cluster_b) & (pl.col("_c2") == cluster_a))
        )
    )
    # Accumulate bridge scores per node (both endpoints)
    if bridge_edges.height == 0:
        top = []
    else:
        s1 = bridge_edges.select(pl.col("uid1").alias("uid"), pl.col("rel_sum2").alias("_w"))
        s2 = bridge_edges.select(pl.col("uid2").alias("uid"), pl.col("rel_sum2").alias("_w"))
        scores = pl.concat([s1, s2]).group_by("uid").agg(pl.col("_w").sum())
        scores = scores.sort("_w", descending=True).head(top_k)
        top = list(zip(scores["uid"].to_list(), scores["_w"].to_list()))

    # Get paper metadata
    results = []
    abs_df = None
    if abstracts_path and Path(abstracts_path).exists():
        abs_df = pl.read_parquet(abstracts_path)
        uid_to_title = dict(zip(abs_df["uid"].to_list(), abs_df["title"].to_list()))
        uid_to_year = dict(zip(
            abs_df["uid"].to_list(),
            abs_df["pubyear"].to_list() if "pubyear" in abs_df.columns else [None] * abs_df.height,
        ))
    else:
        uid_to_title = {}
        uid_to_year = {}

    for uid, score in top:
        results.append({
            "uid": uid,
            "title": uid_to_title.get(uid, ""),
            "year": uid_to_year.get(uid),
            "cluster": uid_to_cluster.get(uid),
            "bridge_score": round(score, 4),
        })

    return results


__all__ = ["build_network_json", "build_term_network_json",
           "build_temporal_snapshots", "find_bridge_papers"]
