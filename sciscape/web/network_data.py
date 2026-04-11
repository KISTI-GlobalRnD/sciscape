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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


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
        all_uids = set(edges["uid1"].to_list()) | set(edges["uid2"].to_list())
        uid_to_cluster = {uid: i for i, uid in enumerate(sorted(all_uids))}

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
            for col in kw_df.columns:
                if "keyword" in col.lower() or "term" in col.lower():
                    keyword_key = col
                    break
            else:
                keyword_key = kw_df.columns[1] if len(kw_df.columns) > 1 else kw_df.columns[0]

            # Top keyword per cluster
            for row in kw_df.iter_rows(named=True):
                cid = row.get(cluster_key)
                kw = row.get(keyword_key, "")
                if cid is not None and cid not in cluster_labels:
                    cluster_labels[int(cid)] = str(kw)
        except Exception:
            pass

    # Compute per-cluster average year and citations if abstracts available
    cluster_years: Dict[int, float] = {}
    cluster_citations: Dict[int, float] = {}
    if mem_df is not None and cluster_col:
        for col in mem_df.columns:
            if col == "pubyear":
                for row in mem_df.iter_rows(named=True):
                    cid = uid_to_cluster.get(row.get("uid"))
                    yr = row.get("pubyear")
                    if cid is not None and yr:
                        cluster_years.setdefault(cid, []).append(yr) if isinstance(cluster_years.get(cid), list) else None
                break

    # Build nodes
    nodes = []
    for cid in sorted(cluster_sizes.keys()):
        size = cluster_sizes[cid]
        nodes.append({
            "id": int(cid),
            "size": size,
            "pct": round(100 * size / n_total, 2) if n_total else 0,
            "label": cluster_labels.get(int(cid), f"C{cid}"),
            "year": 0,
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
    cluster_edge_weights: Dict[tuple, float] = defaultdict(float)

    for row in edges.iter_rows(named=True):
        u1 = row.get("uid1")
        u2 = row.get("uid2")
        w = row.get("rel_sum2", 1.0)
        c1 = uid_to_cluster.get(u1)
        c2 = uid_to_cluster.get(u2)
        if c1 is None or c2 is None or c1 == c2:
            continue
        pair = (min(c1, c2), max(c1, c2))
        cluster_edge_weights[pair] += float(w)

    # Sort by weight, take top
    sorted_edges = sorted(cluster_edge_weights.items(), key=lambda x: -x[1])[:max_edges]

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
        for row in pairs.iter_rows(named=True):
            links.append({
                "child_level": levels[i],
                "parent_level": levels[i + 1],
                "child": int(row[child_col]),
                "parent": int(row["parent"]),
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
    keyword_col = next(
        (c for c in cols if any(k in c.lower() for k in ("keyword", "term", "word"))),
        cols[1] if len(cols) > 1 else cols[0],
    )
    score_col = next(
        (c for c in cols if any(k in c.lower() for k in ("score", "weight", "tfidf"))),
        cols[2] if len(cols) > 2 else None,
    )

    # Collect top-k keywords per cluster
    term_clusters: Dict[str, List[int]] = defaultdict(list)  # term → [cluster_ids]
    term_scores: Dict[str, float] = defaultdict(float)       # term → max score
    term_dominant: Dict[str, int] = {}                        # term → dominant cluster

    cluster_ids = sorted(kw_df[cluster_col].unique().to_list())
    for cid in cluster_ids:
        cluster_kws = kw_df.filter(pl.col(cluster_col) == cid)
        if score_col:
            cluster_kws = cluster_kws.sort(score_col, descending=True)
        top = cluster_kws.head(top_k_per_cluster)
        for row in top.iter_rows(named=True):
            term = str(row[keyword_col])
            score = float(row[score_col]) if score_col else 1.0
            term_clusters[term].append(int(cid))
            if score > term_scores[term]:
                term_scores[term] = score
                term_dominant[term] = int(cid)

    # Select top terms by score
    sorted_terms = sorted(term_scores.keys(), key=lambda t: -term_scores[t])[:max_terms]
    term_set = set(sorted_terms)
    term_to_idx = {t: i for i, t in enumerate(sorted_terms)}

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


__all__ = ["build_network_json", "build_term_network_json"]
