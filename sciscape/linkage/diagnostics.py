"""Diagnostic tools for link-type edge characterization.

Analyze coverage, overlap, complementarity, and structural properties
of different link types (DC, BC, CC, Emb) to guide combination strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Per-type statistics
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EdgeStats:
    """Summary statistics for one edge set."""

    name: str
    n_edges: int
    n_nodes: int
    coverage: float           # fraction of focal nodes with ≥1 edge
    mean_degree: float
    median_degree: float
    max_degree: int
    weight_mean: float
    weight_median: float
    weight_std: float
    weight_p5: float          # 5th percentile
    weight_p95: float         # 95th percentile


def edge_stats(
    edges: pl.DataFrame,
    name: str = "",
    *,
    node_ids: Optional[Set[str]] = None,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
    weight_col: str = "rel_sum2",
) -> EdgeStats:
    """Compute summary statistics for an edge table.

    Parameters
    ----------
    edges : pl.DataFrame
        Edge table.
    name : str
        Label for this edge set.
    node_ids : set, optional
        Full focal node set (for coverage calculation).
        If None, coverage is computed against nodes in edges.
    """
    if edges.height == 0:
        n_total = len(node_ids) if node_ids else 0
        return EdgeStats(
            name=name, n_edges=0, n_nodes=0, coverage=0.0,
            mean_degree=0.0, median_degree=0.0, max_degree=0,
            weight_mean=0.0, weight_median=0.0, weight_std=0.0,
            weight_p5=0.0, weight_p95=0.0,
        )

    all_nodes = pl.concat([edges[uid1_col], edges[uid2_col]]).unique()
    n_in_edges = all_nodes.len()
    n_total = len(node_ids) if node_ids else n_in_edges

    # Degree distribution
    deg = pl.concat([
        edges.select(pl.col(uid1_col).alias("node")),
        edges.select(pl.col(uid2_col).alias("node")),
    ]).group_by("node").len().rename({"len": "degree"})

    degrees = deg["degree"]
    w = edges[weight_col]

    return EdgeStats(
        name=name,
        n_edges=edges.height,
        n_nodes=int(n_in_edges),
        coverage=float(n_in_edges) / n_total if n_total > 0 else 0.0,
        mean_degree=float(degrees.mean()),
        median_degree=float(degrees.median()),
        max_degree=int(degrees.max()),
        weight_mean=float(w.mean()),
        weight_median=float(w.median()),
        weight_std=float(w.std()) if w.len() > 1 else 0.0,
        weight_p5=float(w.quantile(0.05)),
        weight_p95=float(w.quantile(0.95)),
    )


def stats_table(stats_list: list[EdgeStats]) -> pl.DataFrame:
    """Convert a list of EdgeStats to a comparison DataFrame."""
    return pl.DataFrame([
        {
            "name": s.name,
            "edges": s.n_edges,
            "nodes": s.n_nodes,
            "coverage": round(s.coverage, 4),
            "mean_deg": round(s.mean_degree, 1),
            "median_deg": round(s.median_degree, 1),
            "max_deg": s.max_degree,
            "w_mean": round(s.weight_mean, 6),
            "w_median": round(s.weight_median, 6),
            "w_std": round(s.weight_std, 6),
            "w_p5": round(s.weight_p5, 6),
            "w_p95": round(s.weight_p95, 6),
        }
        for s in stats_list
    ])


# ═══════════════════════════════════════════════════════════════════
# Edge overlap analysis
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OverlapResult:
    """Pairwise overlap between two edge sets."""

    name_a: str
    name_b: str
    n_a: int                  # edges in A
    n_b: int                  # edges in B
    n_intersection: int       # edges in both
    n_union: int              # edges in either
    n_only_a: int             # edges only in A
    n_only_b: int             # edges only in B
    jaccard: float            # |A∩B| / |A∪B|
    overlap_coeff: float      # |A∩B| / min(|A|, |B|)
    a_exclusive_frac: float   # |only_A| / |A|
    b_exclusive_frac: float   # |only_B| / |B|


def _canonical_pairs(
    df: pl.DataFrame,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
) -> pl.DataFrame:
    """Extract canonical sorted (min, max) edge pairs, deduplicated."""
    return df.select(
        pl.min_horizontal(uid1_col, uid2_col).alias("_a"),
        pl.max_horizontal(uid1_col, uid2_col).alias("_b"),
    ).unique()


def edge_overlap(
    edges_a: pl.DataFrame,
    edges_b: pl.DataFrame,
    name_a: str = "A",
    name_b: str = "B",
    *,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
) -> OverlapResult:
    """Compute pairwise edge overlap between two edge sets."""
    pairs_a = _canonical_pairs(edges_a, uid1_col, uid2_col)
    pairs_b = _canonical_pairs(edges_b, uid1_col, uid2_col)

    n_a = pairs_a.height
    n_b = pairs_b.height
    n_int = pairs_a.join(pairs_b, on=["_a", "_b"], how="inner").height
    n_union = n_a + n_b - n_int
    n_only_a = n_a - n_int
    n_only_b = n_b - n_int

    return OverlapResult(
        name_a=name_a,
        name_b=name_b,
        n_a=n_a,
        n_b=n_b,
        n_intersection=n_int,
        n_union=n_union,
        n_only_a=n_only_a,
        n_only_b=n_only_b,
        jaccard=n_int / n_union if n_union > 0 else 0.0,
        overlap_coeff=n_int / min(n_a, n_b) if min(n_a, n_b) > 0 else 0.0,
        a_exclusive_frac=n_only_a / n_a if n_a > 0 else 0.0,
        b_exclusive_frac=n_only_b / n_b if n_b > 0 else 0.0,
    )


def overlap_matrix(
    edge_sets: Dict[str, pl.DataFrame],
    *,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
) -> Tuple[list[OverlapResult], pl.DataFrame]:
    """Compute all pairwise overlaps and return a Jaccard matrix.

    Returns
    -------
    results : list of OverlapResult
    jaccard_df : pl.DataFrame
        Square matrix with Jaccard indices.
    """
    names = list(edge_sets.keys())
    pair_sets = {n: _edge_set(df, uid1_col, uid2_col) for n, df in edge_sets.items()}

    results = []
    jaccard = {n: {} for n in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                jaccard[a][b] = 1.0
                continue
            if i > j:
                jaccard[a][b] = jaccard[b][a]
                continue
            sa, sb = pair_sets[a], pair_sets[b]
            n_int = len(sa & sb)
            n_union = len(sa | sb)
            j_val = n_int / n_union if n_union > 0 else 0.0
            jaccard[a][b] = j_val
            results.append(OverlapResult(
                name_a=a, name_b=b,
                n_a=len(sa), n_b=len(sb),
                n_intersection=n_int, n_union=n_union,
                n_only_a=len(sa - sb), n_only_b=len(sb - sa),
                jaccard=j_val,
                overlap_coeff=n_int / min(len(sa), len(sb)) if min(len(sa), len(sb)) > 0 else 0.0,
                a_exclusive_frac=len(sa - sb) / len(sa) if len(sa) > 0 else 0.0,
                b_exclusive_frac=len(sb - sa) / len(sb) if len(sb) > 0 else 0.0,
            ))

    jaccard_df = pl.DataFrame({"name": names, **{n: [jaccard[r][n] for r in names] for n in names}})
    return results, jaccard_df


# ═══════════════════════════════════════════════════════════════════
# Node-level complementarity
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ComplementarityResult:
    """How much unique information each edge type adds."""

    name: str
    n_nodes_total: int
    n_nodes_covered: int
    n_nodes_exclusive: int    # nodes ONLY reachable through this type
    exclusive_frac: float
    n_edges_exclusive: int    # edges ONLY in this type
    edge_exclusive_frac: float


def complementarity_analysis(
    edge_sets: Dict[str, pl.DataFrame],
    *,
    node_ids: Optional[Set[str]] = None,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
) -> list[ComplementarityResult]:
    """Analyze what unique information each link type contributes.

    For each type, compute:
    - Nodes only reachable through this type (no other type connects them)
    - Edges exclusive to this type
    """
    # Build per-type edge sets and node sets
    pair_sets = {}
    node_sets = {}
    for name, df in edge_sets.items():
        pair_sets[name] = _edge_set(df, uid1_col, uid2_col)
        nodes = set(pl.concat([df[uid1_col], df[uid2_col]]).unique().to_list())
        node_sets[name] = nodes

    all_nodes = set()
    for ns in node_sets.values():
        all_nodes |= ns
    if node_ids is not None:
        n_total = len(node_ids)
    else:
        n_total = len(all_nodes)

    results = []
    names = list(edge_sets.keys())

    for name in names:
        # Nodes in this type but no other
        others_nodes = set()
        for other in names:
            if other != name:
                others_nodes |= node_sets[other]
        exclusive_nodes = node_sets[name] - others_nodes

        # Edges in this type but no other
        others_edges = set()
        for other in names:
            if other != name:
                others_edges |= pair_sets[other]
        exclusive_edges = pair_sets[name] - others_edges

        n_covered = len(node_sets[name])
        n_excl = len(exclusive_nodes)
        n_edge_excl = len(exclusive_edges)
        n_edges = len(pair_sets[name])

        results.append(ComplementarityResult(
            name=name,
            n_nodes_total=n_total,
            n_nodes_covered=n_covered,
            n_nodes_exclusive=n_excl,
            exclusive_frac=n_excl / n_total if n_total > 0 else 0.0,
            n_edges_exclusive=n_edge_excl,
            edge_exclusive_frac=n_edge_excl / n_edges if n_edges > 0 else 0.0,
        ))

    return results


def complementarity_table(results: list[ComplementarityResult]) -> pl.DataFrame:
    return pl.DataFrame([
        {
            "name": r.name,
            "nodes_covered": r.n_nodes_covered,
            "nodes_exclusive": r.n_nodes_exclusive,
            "excl_node_%": round(r.exclusive_frac * 100, 2),
            "edges_exclusive": r.n_edges_exclusive,
            "excl_edge_%": round(r.edge_exclusive_frac * 100, 2),
        }
        for r in results
    ])


# ═══════════════════════════════════════════════════════════════════
# Weight correlation between types (for shared edges)
# ═══════════════════════════════════════════════════════════════════

def weight_correlation(
    edges_a: pl.DataFrame,
    edges_b: pl.DataFrame,
    name_a: str = "A",
    name_b: str = "B",
    *,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
    weight_col: str = "rel_sum2",
) -> Dict[str, float]:
    """Compute weight correlation on shared edges between two types.

    Returns Pearson, Spearman, and Kendall correlations, plus
    the fraction of shared edges where types agree on relative ordering.
    """
    # Normalize pair keys
    a_norm = edges_a.with_columns(
        pl.min_horizontal(uid1_col, uid2_col).alias("_lo"),
        pl.max_horizontal(uid1_col, uid2_col).alias("_hi"),
    ).select("_lo", "_hi", pl.col(weight_col).alias("w_a"))

    b_norm = edges_b.with_columns(
        pl.min_horizontal(uid1_col, uid2_col).alias("_lo"),
        pl.max_horizontal(uid1_col, uid2_col).alias("_hi"),
    ).select("_lo", "_hi", pl.col(weight_col).alias("w_b"))

    shared = a_norm.join(b_norm, on=["_lo", "_hi"], how="inner")

    if shared.height < 3:
        return {
            "n_shared": shared.height,
            "pearson": float("nan"),
            "spearman": float("nan"),
        }

    wa = shared["w_a"].to_numpy()
    wb = shared["w_b"].to_numpy()

    from scipy.stats import pearsonr, spearmanr
    pearson, _ = pearsonr(wa, wb)
    spearman, _ = spearmanr(wa, wb)

    return {
        "name_a": name_a,
        "name_b": name_b,
        "n_shared": shared.height,
        "pearson": round(float(pearson), 4),
        "spearman": round(float(spearman), 4),
    }


# ═══════════════════════════════════════════════════════════════════
# Degree-based characterization
# ═══════════════════════════════════════════════════════════════════

def degree_comparison(
    edge_sets: Dict[str, pl.DataFrame],
    *,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
) -> pl.DataFrame:
    """Compare per-node degree across link types.

    Returns a DataFrame with one row per node and a degree column per type.
    Useful for identifying hub vs. peripheral behavior across types.
    """
    all_degs = []
    for name, df in edge_sets.items():
        deg = (
            pl.concat([
                df.select(pl.col(uid1_col).alias("node")),
                df.select(pl.col(uid2_col).alias("node")),
            ])
            .group_by("node").len()
            .rename({"len": f"deg_{name}"})
        )
        all_degs.append(deg)

    result = all_degs[0]
    for d in all_degs[1:]:
        result = result.join(d, on="node", how="full", coalesce=True)

    # Fill nulls with 0 (node not in that type)
    for col in result.columns:
        if col.startswith("deg_"):
            result = result.with_columns(pl.col(col).fill_null(0))

    return result.sort("node")


__all__ = [
    "ComplementarityResult",
    "EdgeStats",
    "OverlapResult",
    "complementarity_analysis",
    "complementarity_table",
    "degree_comparison",
    "edge_overlap",
    "edge_stats",
    "overlap_matrix",
    "stats_table",
    "weight_correlation",
]
