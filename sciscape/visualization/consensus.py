"""Consensus distribution and backbone visualization.

Visualizes multi-layer edge consensus: how many layers agree on each
connection, which connections form the backbone, and how consensus
correlates with cluster structure.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def compute_consensus_stats(
    layers: Dict[str, pl.DataFrame],
    *,
    top_k: int = 30,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
) -> Dict[str, Any]:
    """Compute per-edge consensus statistics across layers.

    Returns a dict with:
    - n_layers_distribution: {1: count, 2: count, ...}
    - per_layer_coverage: {layer_name: n_nodes}
    - overlap_matrix: pairwise overlap counts between layers
    - backbone_edges: edges present in all layers (strongest consensus)
    - total_edges: union edge count
    """
    from ..linkage.filters import filter_top_k

    # Top-k filter each layer
    filtered = {}
    for name, df in layers.items():
        if df.height == 0:
            continue
        if top_k > 0:
            df = filter_top_k(df, top_k, uid1_col=uid1_col, uid2_col=uid2_col)
        filtered[name] = df

    # Build pair → set of layers
    pair_layers: Dict[Tuple[str, str], set] = defaultdict(set)
    for name, df in filtered.items():
        for row in df.iter_rows(named=True):
            pair = (min(row[uid1_col], row[uid2_col]),
                    max(row[uid1_col], row[uid2_col]))
            pair_layers[pair].add(name)

    # N-layers distribution
    n_layers_dist = Counter(len(v) for v in pair_layers.values())

    # Per-layer coverage
    layer_nodes = {}
    for name, df in filtered.items():
        nodes = set(df[uid1_col].to_list()) | set(df[uid2_col].to_list())
        layer_nodes[name] = len(nodes)

    # Pairwise overlap matrix
    layer_names = sorted(filtered.keys())
    overlap = {}
    for i, a in enumerate(layer_names):
        for j, b in enumerate(layer_names):
            if i <= j:
                pairs_a = {(min(r[uid1_col], r[uid2_col]), max(r[uid1_col], r[uid2_col]))
                           for r in filtered[a].iter_rows(named=True)}
                pairs_b = {(min(r[uid1_col], r[uid2_col]), max(r[uid1_col], r[uid2_col]))
                           for r in filtered[b].iter_rows(named=True)}
                overlap[(a, b)] = len(pairs_a & pairs_b)
                overlap[(b, a)] = overlap[(a, b)]

    # Backbone: edges in ALL layers
    n_all = len(filtered)
    backbone = [pair for pair, ls in pair_layers.items() if len(ls) == n_all]

    return {
        "n_layers_distribution": dict(sorted(n_layers_dist.items())),
        "per_layer_coverage": layer_nodes,
        "per_layer_edges": {name: df.height for name, df in filtered.items()},
        "overlap_matrix": {f"{a}_{b}": c for (a, b), c in overlap.items()},
        "overlap_layer_names": layer_names,
        "backbone_size": len(backbone),
        "total_edges": len(pair_layers),
        "n_layers": n_all,
    }


def compute_consensus_vs_cluster(
    layers: Dict[str, pl.DataFrame],
    membership: Dict[str, int],
    *,
    top_k: int = 30,
) -> Dict[str, Any]:
    """Analyze how consensus level correlates with cluster structure.

    For each consensus level (1-layer, 2-layer, ...):
    - What fraction of edges are intra-cluster vs cross-cluster?
    - What's the average weight?

    Returns dict with per-level stats.
    """
    from ..linkage.filters import filter_top_k

    filtered = {}
    for name, df in layers.items():
        if df.height == 0:
            continue
        if top_k > 0:
            df = filter_top_k(df, top_k)
        filtered[name] = df

    # Build pair → n_layers
    pair_nlayers: Dict[Tuple, int] = defaultdict(int)
    for name, df in filtered.items():
        for row in df.iter_rows(named=True):
            pair = (min(row["uid1"], row["uid2"]), max(row["uid1"], row["uid2"]))
            pair_nlayers[pair] += 1

    # Classify intra/cross per consensus level
    level_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {"intra": 0, "cross": 0, "unknown": 0})

    for (u1, u2), nl in pair_nlayers.items():
        c1 = membership.get(u1)
        c2 = membership.get(u2)
        if c1 is None or c2 is None:
            level_stats[nl]["unknown"] += 1
        elif c1 == c2:
            level_stats[nl]["intra"] += 1
        else:
            level_stats[nl]["cross"] += 1

    # Compute ratios
    result = {}
    for nl in sorted(level_stats.keys()):
        s = level_stats[nl]
        total = s["intra"] + s["cross"]
        result[nl] = {
            "n_edges": total + s["unknown"],
            "intra": s["intra"],
            "cross": s["cross"],
            "intra_pct": round(100 * s["intra"] / total, 1) if total else 0,
            "cross_pct": round(100 * s["cross"] / total, 1) if total else 0,
        }

    return result


def format_consensus_report(
    stats: Dict[str, Any],
    cluster_stats: Dict[int, Dict] | None = None,
) -> str:
    """Format consensus stats as a readable text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("Multi-layer Consensus Report")
    lines.append("=" * 60)

    lines.append(f"\nLayers: {stats['n_layers']}")
    lines.append(f"Total edges (union): {stats['total_edges']:,}")
    lines.append(f"Backbone (all-layer): {stats['backbone_size']:,} "
                 f"({100*stats['backbone_size']/max(stats['total_edges'],1):.1f}%)")

    lines.append("\n--- Consensus distribution ---")
    for nl, count in stats["n_layers_distribution"].items():
        pct = 100 * count / stats["total_edges"]
        bar = "█" * int(pct / 2)
        lines.append(f"  {nl}-layer: {count:>10,} ({pct:>5.1f}%) {bar}")

    lines.append("\n--- Per-layer coverage ---")
    for name in sorted(stats["per_layer_coverage"]):
        nodes = stats["per_layer_coverage"][name]
        edges = stats["per_layer_edges"].get(name, 0)
        lines.append(f"  {name:>20}: {nodes:>8,} nodes, {edges:>10,} edges")

    if cluster_stats:
        lines.append("\n--- Consensus vs cluster structure ---")
        lines.append(f"  {'N_layers':>8} {'Edges':>10} {'Intra%':>8} {'Cross%':>8}")
        for nl, s in sorted(cluster_stats.items()):
            lines.append(f"  {nl:>8} {s['n_edges']:>10,} {s['intra_pct']:>7.1f}% {s['cross_pct']:>7.1f}%")

    return "\n".join(lines)


def consensus_to_plotly(
    stats: Dict[str, Any],
    cluster_stats: Dict[int, Dict] | None = None,
) -> Dict[str, Any]:
    """Generate Plotly figure data for consensus visualization.

    Returns dict of figure JSONs for embedding in HTML.
    """
    figures = {}

    # 1. Consensus distribution bar chart
    dist = stats["n_layers_distribution"]
    figures["distribution"] = {
        "data": [{
            "x": [f"{k}-layer" for k in dist],
            "y": list(dist.values()),
            "type": "bar",
            "marker": {"color": ["#94a3b8", "#60a5fa", "#f59e0b", "#10b981"][:len(dist)]},
        }],
        "layout": {
            "title": "Edge Consensus Distribution",
            "xaxis": {"title": "Number of agreeing layers"},
            "yaxis": {"title": "Edge count"},
            "height": 350,
        },
    }

    # 2. Intra/cross ratio per consensus level
    if cluster_stats:
        levels = sorted(cluster_stats.keys())
        figures["intra_cross"] = {
            "data": [
                {
                    "x": [f"{nl}L" for nl in levels],
                    "y": [cluster_stats[nl]["intra_pct"] for nl in levels],
                    "name": "Intra-cluster",
                    "type": "bar",
                    "marker": {"color": "#10b981"},
                },
                {
                    "x": [f"{nl}L" for nl in levels],
                    "y": [cluster_stats[nl]["cross_pct"] for nl in levels],
                    "name": "Cross-cluster",
                    "type": "bar",
                    "marker": {"color": "#ef4444"},
                },
            ],
            "layout": {
                "title": "Intra vs Cross-cluster by Consensus Level",
                "barmode": "stack",
                "yaxis": {"title": "%", "range": [0, 100]},
                "height": 350,
            },
        }

    # 3. Overlap heatmap
    layer_names = stats.get("overlap_layer_names", [])
    if layer_names:
        matrix = []
        for a in layer_names:
            row = []
            for b in layer_names:
                row.append(stats["overlap_matrix"].get(f"{a}_{b}", 0))
            matrix.append(row)

        figures["overlap_heatmap"] = {
            "data": [{
                "z": matrix,
                "x": layer_names,
                "y": layer_names,
                "type": "heatmap",
                "colorscale": "Blues",
            }],
            "layout": {
                "title": "Layer Pairwise Edge Overlap",
                "height": 400,
            },
        }

    return figures


__all__ = [
    "compute_consensus_stats",
    "compute_consensus_vs_cluster",
    "format_consensus_report",
    "consensus_to_plotly",
]
