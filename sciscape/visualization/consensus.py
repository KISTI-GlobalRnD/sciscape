"""Consensus distribution and backbone visualization.

Visualizes multi-layer edge consensus: how many layers agree on each
connection, which connections form the backbone, and how consensus
correlates with cluster structure.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List

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

    # Normalize pairs (lo, hi) and count layers per edge — fully vectorized
    # Step 1: build normalized pair table per layer with layer tag
    tagged_parts = []
    for name, df in filtered.items():
        normed = df.select(
            pl.min_horizontal(uid1_col, uid2_col).alias("_lo"),
            pl.max_horizontal(uid1_col, uid2_col).alias("_hi"),
        ).with_columns(pl.lit(name).alias("_layer"))
        tagged_parts.append(normed)

    if not tagged_parts:
        return {
            "n_layers_distribution": {}, "per_layer_coverage": {},
            "per_layer_edges": {}, "overlap_matrix": {}, "overlap_layer_names": [],
            "backbone_size": 0, "total_edges": 0, "n_layers": 0,
        }

    all_tagged = pl.concat(tagged_parts)

    # N-layers distribution: count distinct layers per pair
    pair_counts = all_tagged.group_by(["_lo", "_hi"]).agg(
        pl.col("_layer").n_unique().alias("_n_layers")
    )
    n_layers_dist = dict(
        pair_counts.group_by("_n_layers").len()
        .sort("_n_layers")
        .iter_rows()
    )

    # Per-layer coverage
    layer_nodes = {}
    for name, df in filtered.items():
        layer_nodes[name] = pl.concat([df[uid1_col], df[uid2_col]]).n_unique()

    # Pairwise overlap matrix: precompute pair sets once per layer
    layer_names = sorted(filtered.keys())
    pair_sets: Dict[str, set] = {}
    for name, df in filtered.items():
        normed = df.select(
            pl.min_horizontal(uid1_col, uid2_col).alias("_lo"),
            pl.max_horizontal(uid1_col, uid2_col).alias("_hi"),
        )
        pair_sets[name] = set(zip(normed["_lo"].to_list(), normed["_hi"].to_list()))

    overlap = {}
    for i, a in enumerate(layer_names):
        for j, b in enumerate(layer_names):
            if i <= j:
                overlap[(a, b)] = len(pair_sets[a] & pair_sets[b])
                overlap[(b, a)] = overlap[(a, b)]

    # Backbone: edges in ALL layers
    n_all = len(filtered)
    backbone_count = pair_counts.filter(pl.col("_n_layers") == n_all).height

    return {
        "n_layers_distribution": dict(sorted(n_layers_dist.items())),
        "per_layer_coverage": layer_nodes,
        "per_layer_edges": {name: df.height for name, df in filtered.items()},
        "overlap_matrix": {f"{a}_{b}": c for (a, b), c in overlap.items()},
        "overlap_layer_names": layer_names,
        "backbone_size": backbone_count,
        "total_edges": pair_counts.height,
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

    # Build pair → n_layers (vectorized)
    tagged_parts = []
    for name, df in filtered.items():
        normed = df.select(
            pl.min_horizontal("uid1", "uid2").alias("_lo"),
            pl.max_horizontal("uid1", "uid2").alias("_hi"),
        )
        tagged_parts.append(normed)

    if not tagged_parts:
        return {}

    pair_nlayers_df = pl.concat(tagged_parts).group_by(["_lo", "_hi"]).len(name="_nl")
    pair_nlayers: Dict[tuple, int] = {
        (r[0], r[1]): r[2] for r in pair_nlayers_df.iter_rows()
    }

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
