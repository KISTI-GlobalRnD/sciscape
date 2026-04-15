"""Temporal tracking of cluster structure evolution.

Tracks how clusters grow, split, merge, and emerge over time
by computing rolling-window memberships and comparing with AMI.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def compute_temporal_snapshots(
    edges: pl.DataFrame,
    year_map: Dict[str, int],
    membership: Dict[str, int] | np.ndarray | None = None,
    *,
    window_years: int = 5,
    step_years: int = 1,
    year_range: Tuple[int, int] | None = None,
) -> List[Dict[str, Any]]:
    """Compute cluster structure at each time window.

    For each window [t, t+window_years):
    - Filter edges to papers published in that window
    - If membership provided: compute cluster composition at that time
    - Count active clusters, size distribution

    Parameters
    ----------
    edges : pl.DataFrame
        uid1, uid2, rel_sum2.
    year_map : dict
        {uid: publication_year}.
    membership : dict or array, optional
        Cluster assignment (from full-period clustering).
    window_years : int
        Window width in years.
    step_years : int
        Step between windows.

    Returns
    -------
    list of dicts, one per time window.
    """
    years = sorted(set(y for y in year_map.values() if y and y > 0))
    if not years:
        return []

    if year_range:
        yr_min, yr_max = year_range
    else:
        yr_min, yr_max = min(years), max(years)

    # Convert membership to dict if array
    if isinstance(membership, np.ndarray):
        uids = sorted(year_map.keys())
        membership = {u: int(membership[i]) for i, u in enumerate(uids) if i < len(membership)}

    snapshots = []
    for start in range(yr_min, yr_max - window_years + 2, step_years):
        end = start + window_years
        # Papers in window
        window_uids = {u for u, y in year_map.items() if y and start <= y < end}
        if not window_uids:
            continue

        # Edges in window
        window_edges = edges.filter(
            pl.col("uid1").is_in(list(window_uids)) & pl.col("uid2").is_in(list(window_uids))
        )

        snap = {
            "start": start,
            "end": end,
            "n_papers": len(window_uids),
            "n_edges": window_edges.height,
        }

        # Cluster composition
        if membership:
            cluster_counts = Counter()
            for u in window_uids:
                cid = membership.get(u)
                if cid is not None:
                    cluster_counts[cid] += 1

            snap["n_active_clusters"] = len(cluster_counts)
            snap["top5_clusters"] = [
                {"cluster": cid, "count": cnt}
                for cid, cnt in cluster_counts.most_common(5)
            ]
            snap["cluster_sizes"] = dict(cluster_counts)

        snapshots.append(snap)

    log.info("temporal_snapshots: %d windows (step=%d, width=%d)",
             len(snapshots), step_years, window_years)
    return snapshots


def compute_cluster_growth(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Track per-cluster growth/decline across time windows.

    Returns {cluster_id: [{year, count}, ...]}.
    """
    cluster_series: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for snap in snapshots:
        sizes = snap.get("cluster_sizes", {})
        for cid, cnt in sizes.items():
            cluster_series[cid].append({
                "year": snap["start"],
                "count": cnt,
            })

    return dict(cluster_series)


def detect_emerging_clusters(
    snapshots: List[Dict[str, Any]],
    *,
    min_growth_rate: float = 2.0,
    min_final_size: int = 50,
) -> List[Dict[str, Any]]:
    """Find clusters that grew rapidly in recent windows.

    Parameters
    ----------
    min_growth_rate : float
        Minimum ratio (last / first window count) to be considered "emerging".
    min_final_size : int
        Minimum papers in the latest window.

    Returns
    -------
    list of emerging cluster dicts.
    """
    if len(snapshots) < 2:
        return []

    growth = compute_cluster_growth(snapshots)
    emerging = []

    for cid, series in growth.items():
        if len(series) < 2:
            continue
        first = series[0]["count"]
        last = series[-1]["count"]
        if first == 0 or last < min_final_size:
            continue
        rate = last / first
        if rate >= min_growth_rate:
            emerging.append({
                "cluster": cid,
                "growth_rate": round(rate, 2),
                "first_count": first,
                "last_count": last,
                "first_year": series[0]["year"],
                "last_year": series[-1]["year"],
            })

    emerging.sort(key=lambda x: -x["growth_rate"])
    return emerging


def temporal_to_plotly(
    snapshots: List[Dict[str, Any]],
    *,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Generate Plotly figures for temporal tracking.

    Returns dict of figure JSONs.
    """
    figures = {}

    years = [s["start"] for s in snapshots]

    # 1. Paper + edge count over time
    figures["activity"] = {
        "data": [
            {"x": years, "y": [s["n_papers"] for s in snapshots],
             "name": "Papers", "type": "scatter", "mode": "lines+markers"},
            {"x": years, "y": [s["n_edges"] for s in snapshots],
             "name": "Edges", "type": "scatter", "mode": "lines+markers", "yaxis": "y2"},
        ],
        "layout": {
            "title": "Research Activity Over Time",
            "xaxis": {"title": "Window start year"},
            "yaxis": {"title": "Papers"},
            "yaxis2": {"title": "Edges", "overlaying": "y", "side": "right"},
            "height": 350,
        },
    }

    # 2. Active cluster count
    if snapshots and "n_active_clusters" in snapshots[0]:
        figures["active_clusters"] = {
            "data": [{
                "x": years,
                "y": [s.get("n_active_clusters", 0) for s in snapshots],
                "type": "scatter", "mode": "lines+markers",
                "line": {"color": "#2563eb"},
            }],
            "layout": {
                "title": "Active Clusters Over Time",
                "xaxis": {"title": "Window start year"},
                "yaxis": {"title": "Number of active clusters"},
                "height": 300,
            },
        }

    # 3. Top cluster growth curves
    growth = compute_cluster_growth(snapshots)
    if growth:
        # Pick top_n by final size
        final_sizes = {cid: series[-1]["count"] for cid, series in growth.items() if series}
        top_clusters = sorted(final_sizes, key=lambda c: -final_sizes[c])[:top_n]

        traces = []
        colors = ['#2563eb','#dc2626','#059669','#d97706','#7c3aed',
                  '#db2777','#0891b2','#65a30d','#ea580c','#4f46e5']
        for i, cid in enumerate(top_clusters):
            series = growth[cid]
            traces.append({
                "x": [s["year"] for s in series],
                "y": [s["count"] for s in series],
                "name": f"C{cid}",
                "type": "scatter", "mode": "lines",
                "line": {"color": colors[i % len(colors)]},
            })

        figures["growth_curves"] = {
            "data": traces,
            "layout": {
                "title": f"Top {top_n} Cluster Growth",
                "xaxis": {"title": "Year"},
                "yaxis": {"title": "Papers in cluster"},
                "height": 400,
            },
        }

    return figures


__all__ = [
    "compute_temporal_snapshots",
    "compute_cluster_growth",
    "detect_emerging_clusters",
    "temporal_to_plotly",
]
