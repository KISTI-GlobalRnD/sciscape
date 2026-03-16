"""CPM-critical dendrogram construction via sparse average-linkage HAC.

Thin Python wrapper around the ``cpm_dendro`` Rust crate.  Accepts an
:class:`igraph.Graph` and returns a scipy-compatible linkage matrix.

Usage
-----
>>> from sciscape.clustering.dendrogram import build_dendrogram
>>> from sciscape.clustering.constrained_cut import constrained_cut
>>> linkage = build_dendrogram(graph, triadic=True)
>>> result = constrained_cut(linkage, min_size=1000)

The Rust core runs exact average-linkage HAC in Õ(n√m) time, producing
a complete binary dendrogram where merge heights equal CPM critical
resolutions: γ*(A,B) = e_AB / (|A|·|B|).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import igraph as ig
except ImportError:
    ig = None  # type: ignore[assignment]


def build_dendrogram(
    graph: "ig.Graph",
    *,
    triadic: bool = False,
    weight_attr: str = "weight",
) -> np.ndarray:
    """Build a CPM-critical dendrogram from an igraph weighted graph.

    Parameters
    ----------
    graph : igraph.Graph
        Undirected weighted graph.
    triadic : bool, default False
        If True, reweight edges by triadic closure before HAC:
        ``w'(i,j) = w(i,j) * (1 + |common_neighbors(i,j)|)``.
    weight_attr : str, default "weight"
        Edge attribute name for weights.

    Returns
    -------
    linkage : numpy.ndarray, shape (n-1, 4)
        Scipy-compatible linkage matrix ``[left, right, height, size]``.
        Rows are ordered by merge sequence (first merge = highest density).
    """
    try:
        import cpm_dendro
    except ImportError:
        raise ImportError(
            "cpm_dendro Rust extension not found. "
            "Build it with: cd cpm-dendro && maturin develop --release"
        ) from None

    n = graph.vcount()
    if n == 0:
        raise ValueError("Graph has no vertices")
    if n == 1:
        return np.empty((0, 4), dtype=np.float64)

    # Extract edge list and weights
    edge_list = graph.get_edgelist()
    sources = np.array([e[0] for e in edge_list], dtype=np.uint32)
    targets = np.array([e[1] for e in edge_list], dtype=np.uint32)

    if weight_attr in graph.es.attributes():
        weights = np.array(graph.es[weight_attr], dtype=np.float64)
    else:
        weights = np.ones(len(edge_list), dtype=np.float64)

    logger.info(
        "Building CPM-critical dendrogram: %d nodes, %d edges, triadic=%s",
        n, len(edge_list), triadic,
    )

    # Call Rust implementation
    linkage = cpm_dendro.build_cpm_dendrogram(n, sources, targets, weights, triadic)

    logger.info(
        "Dendrogram complete: %d merges, height range [%.6f, %.6f]",
        len(linkage),
        linkage[-1, 2] if len(linkage) > 0 else 0,
        linkage[0, 2] if len(linkage) > 0 else 0,
    )

    return linkage


__all__ = ["build_dendrogram"]
