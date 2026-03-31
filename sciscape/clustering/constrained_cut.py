"""Size-constrained optimal cut on a binary dendrogram.

Given a linkage matrix (similarity format, non-increasing heights) and a
minimum cluster size *k*, find the partition that **maximizes the number of
clusters** subject to every cluster having at least *k* nodes.  Ties in
cluster count are broken by **total stability** (sum of persistence values).

**Important**: This module expects similarity linkage where merge heights are
non-increasing (e.g., CPM density).  Persistence = γ_birth − γ_death is
positive under this convention.  Do NOT pass standard scipy distance linkage
(non-decreasing heights) — persistence values would be negative and
tie-breaking would produce wrong results.

The algorithm is a single bottom-up pass over the dendrogram tree — O(n) time,
O(n) space — and is provably optimal for the count-maximisation objective.

Theory
------
At each internal node *v* with children *l*, *r*:

* **Split** if both subtrees can independently form valid partitions
  (opt(l).count ≥ 1 AND opt(r).count ≥ 1).  Yields opt(l) + opt(r) clusters.
* **Keep** the entire subtree as one cluster if size(v) ≥ k.  Yields 1 cluster.
* Splitting always dominates keeping when valid (≥ 2 vs 1), so the greedy
  rule is optimal.

References
----------
* Buchin & Selbach (2022) for the general (l, u)-partition on trees.
* Mauduit & Simonetto (2024) for constrained hierarchical cutting framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CutResult:
    """Result of a size-constrained optimal cut on a dendrogram."""

    partition: List[Set[int]]
    """List of clusters, each a set of original leaf node indices."""

    membership: np.ndarray
    """Array of length *n* mapping each leaf node to its cluster ID (0-based)."""

    n_clusters: int
    """Number of clusters in the partition."""

    total_stability: float
    """Sum of persistence (γ_birth − γ_death) across all cut clusters."""

    cut_nodes: List[int]
    """Dendrogram node IDs (using scipy's internal-node indexing) selected by
    the cut.  Leaves use indices ``0..n-1``; internal nodes ``n..2n-2``."""

    feasible: bool
    """Whether a valid partition satisfying min_size was found.
    If False, the result is a single-cluster fallback that may violate min_size."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _DPState:
    """DP state at a single dendrogram node."""
    count: int          # max achievable cluster count in this subtree
    stability: float    # total persistence (tie-break)
    feasible: bool      # whether any valid partition exists for this subtree
    action: str         # "keep" | "split" | "infeasible"


def _build_parent_height(linkage: np.ndarray, n_leaves: int) -> np.ndarray:
    """Map each node to the merge height of its parent (γ_death).

    Returns numpy array of length (n_leaves + n_internal), indexed by node ID.
    The root has γ_death = 0.
    """
    n_internal = len(linkage)
    n_total = n_leaves + n_internal
    parent_height = np.zeros(n_total, dtype=np.float64)
    for row_idx in range(n_internal):
        left = int(linkage[row_idx, 0])
        right = int(linkage[row_idx, 1])
        height = linkage[row_idx, 2]
        parent_height[left] = height
        parent_height[right] = height
    # Root node has no parent → γ_death = 0 (already initialized to 0)
    return parent_height


def _collect_leaves_iterative(
    node_idx: int,
    linkage: np.ndarray,
    n_leaves: int,
) -> Set[int]:
    """Collect all leaf indices under a dendrogram node (iterative)."""
    leaves: Set[int] = set()
    stack = [node_idx]
    while stack:
        nid = stack.pop()
        if nid < n_leaves:
            leaves.add(nid)
        else:
            row = nid - n_leaves
            stack.append(int(linkage[row, 0]))
            stack.append(int(linkage[row, 1]))
    return leaves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def constrained_cut(
    linkage: np.ndarray,
    min_size: int,
    *,
    leaf_sizes: Optional[np.ndarray] = None,
    n_leaves: Optional[int] = None,
) -> CutResult:
    """Find the partition maximising cluster count with minimum size constraint.

    Parameters
    ----------
    linkage : np.ndarray, shape (n-1, 4)
        **Similarity** linkage matrix (non-increasing heights).
        Each row ``[left, right, height, size]``.
        Row 0 = first merge = highest density.  This is the format produced
        by ``build_dendrogram(as_distance=False)``.
        Do NOT pass scipy distance linkage (non-decreasing heights).
    min_size : int
        Every cluster in the output must have at least this many leaf nodes
        (or original nodes when ``leaf_sizes`` is provided).
    leaf_sizes : np.ndarray or None, optional
        Size of each leaf node.  Pass this when the dendrogram was built on
        a **contracted graph** (supernodes) so that the size constraint
        refers to original node counts, not supernode counts.  Shape
        ``(n_leaves,)`` with integer dtype.  If None, every leaf has size 1.
    n_leaves : int, optional
        Number of original data points (leaves).  If *None*, inferred as
        ``len(linkage) + 1``.

    Returns
    -------
    CutResult
        Optimal partition with cluster count, stability, and node assignments.
        Check ``result.feasible`` to determine if the min_size constraint
        was satisfiable.
    """
    if linkage.ndim != 2 or linkage.shape[1] != 4:
        raise ValueError(f"linkage must be (n-1, 4), got {linkage.shape}")

    if n_leaves is None:
        n_leaves = len(linkage) + 1
    n_internal = len(linkage)
    n_total = n_leaves + n_internal  # total nodes in tree

    if min_size < 1:
        raise ValueError(f"min_size must be ≥ 1, got {min_size}")

    # --- Precompute subtree sizes and parent heights ---
    subtree_size = np.zeros(n_total, dtype=np.int64)
    if leaf_sizes is not None:
        if len(leaf_sizes) != n_leaves:
            raise ValueError(
                f"leaf_sizes length ({len(leaf_sizes)}) must equal "
                f"n_leaves ({n_leaves})"
            )
        for i in range(n_leaves):
            subtree_size[i] = int(leaf_sizes[i])
    else:
        for i in range(n_leaves):
            subtree_size[i] = 1
    for row_idx in range(n_internal):
        node_id = n_leaves + row_idx
        subtree_size[node_id] = int(linkage[row_idx, 3])

    parent_height = _build_parent_height(linkage, n_leaves)

    # --- Bottom-up DP ---
    dp = [_DPState(0, 0.0, False, "infeasible") for _ in range(n_total)]

    # Leaves
    for i in range(n_leaves):
        if subtree_size[i] >= min_size:
            dp[i] = _DPState(1, 0.0, True, "keep")
        else:
            dp[i] = _DPState(0, 0.0, False, "infeasible")

    # Internal nodes (bottom-up: children always have smaller IDs than parents)
    for row_idx in range(n_internal):
        node_id = n_leaves + row_idx
        left = int(linkage[row_idx, 0])
        right = int(linkage[row_idx, 1])
        size = int(subtree_size[node_id])

        left_state = dp[left]
        right_state = dp[right]

        # Option 1: Keep as single cluster
        if size >= min_size:
            gamma_birth = linkage[row_idx, 2]
            gamma_death = parent_height[node_id]
            node_persistence = gamma_birth - gamma_death
            keep = _DPState(1, node_persistence, True, "keep")
        else:
            keep = _DPState(0, 0.0, False, "infeasible")

        # Option 2: Split into children's partitions
        if left_state.feasible and right_state.feasible:
            split_count = left_state.count + right_state.count
            split_stability = left_state.stability + right_state.stability
            split = _DPState(split_count, split_stability, True, "split")
        else:
            split = _DPState(0, 0.0, False, "infeasible")

        # Choose best: lexicographic (count, stability)
        if split.feasible and keep.feasible:
            if split.count > keep.count:
                dp[node_id] = split
            elif split.count == keep.count and split.stability > keep.stability:
                dp[node_id] = split
            elif split.count == keep.count and split.stability == keep.stability:
                # Prefer split (more granular)
                dp[node_id] = split
            else:
                dp[node_id] = keep
        elif split.feasible:
            dp[node_id] = split
        elif keep.feasible:
            dp[node_id] = keep
        else:
            dp[node_id] = _DPState(0, 0.0, False, "infeasible")

    # --- Traceback: collect cut nodes ---
    root_id = n_leaves + n_internal - 1
    root_state = dp[root_id]

    if not root_state.feasible:
        # Entire graph is one cluster (can't satisfy min_size with split)
        all_leaves = set(range(n_leaves))
        membership = np.zeros(n_leaves, dtype=np.int64)
        return CutResult(
            partition=[all_leaves],
            membership=membership,
            n_clusters=1,
            total_stability=0.0,
            cut_nodes=[root_id],
            feasible=False,
        )

    # Traceback via iterative DFS
    cut_nodes: List[int] = []
    stack = [root_id]
    while stack:
        nid = stack.pop()
        state = dp[nid]
        if state.action == "keep":
            cut_nodes.append(nid)
        elif state.action == "split":
            if nid < n_leaves:
                # Leaf chosen as split — means it's a valid singleton cluster
                cut_nodes.append(nid)
            else:
                row = nid - n_leaves
                stack.append(int(linkage[row, 0]))
                stack.append(int(linkage[row, 1]))
        else:
            # infeasible — shouldn't reach here from a feasible root
            cut_nodes.append(nid)

    # --- Build partition from cut nodes ---
    partition: List[Set[int]] = []
    membership = np.full(n_leaves, -1, dtype=np.int64)

    for cluster_id, cut_node in enumerate(cut_nodes):
        leaves = _collect_leaves_iterative(cut_node, linkage, n_leaves)
        partition.append(leaves)
        for leaf in leaves:
            membership[leaf] = cluster_id

    # Total stability
    total_stability = 0.0
    for cut_node in cut_nodes:
        if cut_node < n_leaves:
            total_stability += 0.0
        else:
            row = cut_node - n_leaves
            gamma_birth = linkage[row, 2]
            gamma_death = parent_height[cut_node]
            total_stability += gamma_birth - gamma_death

    return CutResult(
        partition=partition,
        membership=membership,
        n_clusters=len(partition),
        total_stability=total_stability,
        cut_nodes=cut_nodes,
        feasible=True,
    )


__all__ = ["CutResult", "constrained_cut"]
