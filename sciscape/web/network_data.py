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


__all__ = ["build_network_json"]
