"""Greedy CPM-density dendrogram construction via sparse average-linkage HAC.

Thin Python wrapper around the ``cpm_dendro`` Rust crate.  Accepts an
:class:`igraph.Graph` and returns a similarity linkage matrix (non-increasing
merge heights representing pairwise density).

**Important**: This is a greedy approximation. The set of tree-cut partitions
does not necessarily contain the global CPM optimum for every γ.

Usage
-----
>>> from sciscape.clustering.dendrogram import build_dendrogram
>>> from sciscape.clustering.constrained_cut import constrained_cut
>>> linkage = build_dendrogram(graph, mode="cpm")
>>> result = constrained_cut(linkage, min_size=1000)

Merge heights are **similarity** (density, non-increasing). For scipy-compatible
distance linkage (non-decreasing), use ``as_distance=True``.

Modes
-----
- ``"cpm"`` (default): inter-cluster CPM density ρ(A,B) = e_AB / (|A|·|B|)
- ``"triadic_cpm"``: triadic closure preprocessing, then CPM density.
- ``"internal_density"``: merged internal density ρ(A∪B) = (e_A+e_B+e_AB)/C(|A|+|B|,2).
  Heights are internal density of the merged cluster, NOT CPM thresholds.
- ``"triadic_internal_density"``: triadic reweighting + internal density.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import igraph as ig
except ImportError:
    ig = None  # type: ignore[assignment]


def build_dendrogram(
    graph: "ig.Graph",
    *,
    mode: str = "cpm",
    weight_attr: str = "weight",
    node_sizes: "np.ndarray | None" = None,
    as_distance: bool = False,
) -> np.ndarray:
    """Build a greedy CPM-density dendrogram from an igraph weighted graph.

    Parameters
    ----------
    graph : igraph.Graph
        Undirected, non-negative-weighted graph.  Should be the giant
        connected component (GCC).  Disconnected graphs trigger a warning.
        Duplicate edges are accumulated (weights summed).
    mode : str, default "cpm"
        Scoring mode:
        - ``"cpm"``: inter-cluster CPM density ρ(A,B) = e_AB / (|A|·|B|)
        - ``"triadic_cpm"``: triadic reweighting + CPM density
        - ``"internal_density"``: merged internal density ρ(A∪B)
        - ``"triadic_internal_density"``: triadic reweighting + internal density
    weight_attr : str, default "weight"
        Edge attribute name for weights.
    node_sizes : numpy.ndarray or None, default None
        Initial sizes for each leaf node.  Pass this when building a
        dendrogram on a **contracted graph** (supernodes) so that CPM
        density uses the true original node counts.  Shape ``(n,)`` with
        dtype castable to ``uint64``.  All values must be >= 1.
        If None, every leaf has size 1 (standard HAC on individual nodes).
    as_distance : bool, default False
        If True, return scipy-compatible distance linkage (non-decreasing).
        If False (default), return similarity linkage (non-increasing).

    Returns
    -------
    linkage : numpy.ndarray, shape (n-1, 4)
        Linkage matrix ``[left, right, height, size]``.
        Rows are ordered by merge sequence (first merge = highest density).

    Raises
    ------
    ValueError
        If graph has no vertices, is directed, has negative/NaN weights,
        or mode is unknown.
    ImportError
        If cpm_dendro Rust extension is not installed.
    """
    try:
        import cpm_dendro
    except ImportError:
        raise ImportError(
            "cpm_dendro Rust extension not found. "
            "Build it with: cd cpm-dendro && maturin develop --release"
        ) from None

    _valid_modes = ("cpm", "triadic_cpm", "internal_density", "triadic_internal_density")
    if mode not in _valid_modes:
        raise ValueError(
            f"Unknown mode '{mode}'. Supported: {_valid_modes}"
        )

    n = graph.vcount()
    if n == 0:
        raise ValueError("Graph has no vertices")
    if graph.is_directed():
        raise ValueError("Graph must be undirected")
    if n == 1:
        return np.empty((0, 4), dtype=np.float64)

    # Extract edge list and weights
    edge_list = graph.get_edgelist()
    if len(edge_list) > 0:
        edges_arr = np.array(edge_list, dtype=np.uint32)
        sources = np.ascontiguousarray(edges_arr[:, 0])
        targets = np.ascontiguousarray(edges_arr[:, 1])
    else:
        sources = np.array([], dtype=np.uint32)
        targets = np.array([], dtype=np.uint32)

    if weight_attr in graph.es.attributes():
        weights = np.array(graph.es[weight_attr], dtype=np.float64)
    else:
        weights = np.ones(len(edge_list), dtype=np.float64)

    # Validate weights (Rust also validates, but better error messages here)
    if np.any(~np.isfinite(weights)):
        raise ValueError("Edge weights must be finite (no NaN or Inf)")
    if np.any(weights < 0):
        raise ValueError("Edge weights must be non-negative")

    # Warn if graph is disconnected (expects GCC input)
    if not graph.is_connected():
        import warnings
        warnings.warn(
            "Graph is not connected. The dendrogram will contain density-0 "
            "merges between components that have no community semantics. "
            "Consider extracting the giant connected component first.",
            UserWarning,
            stacklevel=2,
        )

    logger.info(
        "Building greedy CPM-density dendrogram: %d nodes, %d edges, mode=%s",
        n, len(edge_list), mode,
    )

    # Prepare node_sizes for Rust
    ns_arr = None
    if node_sizes is not None:
        ns_arr = np.ascontiguousarray(node_sizes, dtype=np.uint64)
        if ns_arr.shape != (n,):
            raise ValueError(
                f"node_sizes must have shape ({n},), got {ns_arr.shape}"
            )

    # Call Rust implementation
    linkage = cpm_dendro.build_cpm_dendrogram(
        n, sources, targets, weights, ns_arr, mode, as_distance
    )

    logger.info(
        "Dendrogram complete: %d merges, height range [%.6f, %.6f]",
        len(linkage),
        linkage[-1, 2] if len(linkage) > 0 else 0,
        linkage[0, 2] if len(linkage) > 0 else 0,
    )

    return linkage


__all__ = ["build_dendrogram"]
