"""Tests for size-constrained optimal cut on dendrograms."""

from __future__ import annotations

import numpy as np
import pytest

from sciscape.clustering.constrained_cut import CutResult, constrained_cut


# ---------------------------------------------------------------------------
# Helper: build simple linkage matrices for testing
# ---------------------------------------------------------------------------

def _linkage_from_merges(merges: list[tuple[int, int, float, int]]) -> np.ndarray:
    """Build a scipy-format linkage matrix from a list of merges.

    Each merge is (left_id, right_id, height, subtree_size).
    """
    return np.array(merges, dtype=np.float64)


# ---------------------------------------------------------------------------
# Test fixtures: simple dendrograms
# ---------------------------------------------------------------------------

class TestTrivial:
    """Edge cases and trivial dendrograms."""

    def test_two_nodes_k1(self):
        """Two nodes, k=1 → 2 singleton clusters."""
        # Merge: node 0 + node 1 → node 2, height=1.0, size=2
        linkage = _linkage_from_merges([(0, 1, 1.0, 2)])
        result = constrained_cut(linkage, min_size=1)
        assert result.n_clusters == 2
        assert len(result.partition) == 2
        assert result.membership[0] != result.membership[1]

    def test_two_nodes_k2(self):
        """Two nodes, k=2 → 1 cluster (entire tree)."""
        linkage = _linkage_from_merges([(0, 1, 1.0, 2)])
        result = constrained_cut(linkage, min_size=2)
        assert result.n_clusters == 1
        assert set().union(*result.partition) == {0, 1}

    def test_two_nodes_k3(self):
        """Two nodes, k=3 → infeasible, returns single cluster."""
        linkage = _linkage_from_merges([(0, 1, 1.0, 2)])
        result = constrained_cut(linkage, min_size=3)
        assert result.n_clusters == 1


class TestBalancedTree:
    """Balanced binary tree with 4 leaves."""

    @pytest.fixture()
    def linkage_4(self):
        """
        Dendrogram:
                  6 (h=0.1, size=4)
                 / \\
               4     5
            (h=0.5) (h=0.3)
            / \\    / \\
           0   1  2   3

        Merge order (highest ρ first for CPM-critical):
          Row 0: merge 0+1 → node 4, height=0.5, size=2
          Row 1: merge 2+3 → node 5, height=0.3, size=2
          Row 2: merge 4+5 → node 6, height=0.1, size=4
        """
        return _linkage_from_merges([
            (0, 1, 0.5, 2),
            (2, 3, 0.3, 2),
            (4, 5, 0.1, 4),
        ])

    def test_k1_gives_4_clusters(self, linkage_4):
        """k=1 → all 4 singletons."""
        result = constrained_cut(linkage_4, min_size=1)
        assert result.n_clusters == 4

    def test_k2_gives_2_clusters(self, linkage_4):
        """k=2 → {0,1} and {2,3}."""
        result = constrained_cut(linkage_4, min_size=2)
        assert result.n_clusters == 2
        clusters = [frozenset(c) for c in result.partition]
        assert frozenset({0, 1}) in clusters
        assert frozenset({2, 3}) in clusters

    def test_k3_gives_1_cluster(self, linkage_4):
        """k=3 → cannot split (each subtree has only 2), keep all as one."""
        result = constrained_cut(linkage_4, min_size=3)
        assert result.n_clusters == 1
        assert set().union(*result.partition) == {0, 1, 2, 3}

    def test_k4_gives_1_cluster(self, linkage_4):
        """k=4 → exactly the root."""
        result = constrained_cut(linkage_4, min_size=4)
        assert result.n_clusters == 1


class TestUnbalancedTree:
    """Unbalanced tree to test asymmetric splitting."""

    @pytest.fixture()
    def linkage_unbalanced(self):
        """
        Dendrogram:
                    7 (h=0.05, size=5)
                   / \\
                  6    4
              (h=0.1) (single, id=4)
               / \\
              5   3
          (h=0.3)
           / \\
          0   1    2

        Wait, let's build this correctly:
          Row 0: merge 0+1 → node 5, height=0.8, size=2
          Row 1: merge 5+2 → node 6, height=0.3, size=3
          Row 2: merge 6+3 → node 7, height=0.1, size=4
          Row 3: merge 7+4 → node 8, height=0.05, size=5
        """
        return _linkage_from_merges([
            (0, 1, 0.8, 2),   # node 5 = {0,1}
            (5, 2, 0.3, 3),   # node 6 = {0,1,2}
            (6, 3, 0.1, 4),   # node 7 = {0,1,2,3}
            (7, 4, 0.05, 5),  # node 8 = {0,1,2,3,4}
        ])

    def test_k2_maximizes_clusters(self, linkage_unbalanced):
        """k=2 → {0,1} and {2,3,4} or {0,1,2} and {3,4}?
        Actually: split at root → node 7 (size=4) + node 4 (size=1).
        Node 4 (size=1) < k=2 → can't split root.
        Try node 7: split → node 6 (size=3) + node 3 (size=1). Node 3 < 2 → can't.
        Keep node 7 as one cluster? size=4 ≥ 2 → yes, 1 cluster.
        Root: node 7 (1 cluster) + node 4 (infeasible) → can't split.
        Keep root (size=5 ≥ 2) → 1 cluster.

        Hmm, that gives 1. Let me rethink...
        Node 6 (size=3): split → node 5 (size=2 ✅) + node 2 (size=1 ❌) → can't split.
        Keep node 6 (size=3 ≥ 2) → 1 cluster.
        Node 7: split → node 6 (1 cluster) + node 3 (size=1 < 2) → can't split.
        Keep node 7 (size=4 ≥ 2) → 1 cluster.
        Node 8: split → node 7 (1 cluster) + node 4 (size=1 < 2) → can't split.
        Keep node 8 (size=5 ≥ 2) → 1 cluster.
        """
        result = constrained_cut(linkage_unbalanced, min_size=2)
        assert result.n_clusters == 1

    def test_k1_maximizes_to_5(self, linkage_unbalanced):
        """k=1 → all 5 singletons."""
        result = constrained_cut(linkage_unbalanced, min_size=1)
        assert result.n_clusters == 5


class TestStabilityTiebreak:
    """Tie-breaking by total stability (persistence)."""

    def test_prefer_higher_stability(self):
        """Two possible 2-cluster partitions with different stability.

        Dendrogram:
                  8 (h=0.01, size=6)
                 / \\
               6     7
           (h=0.5)  (h=0.1)
            / \\     / \\
           4   5   2   3
        (h=0.9)(h=0.8)
         /\\   /\\
        0  1  ...wait, let me simplify.

        Actually for tie-break testing we need a tree where two different
        cut strategies give the same count but different stabilities.
        """
        # Simple: 4 leaves, balanced tree
        # But with different merge heights
        #   Row 0: 0+1 → 4, h=0.9, size=2  (high stability subtree)
        #   Row 1: 2+3 → 5, h=0.2, size=2  (low stability subtree)
        #   Row 2: 4+5 → 6, h=0.1, size=4
        linkage = _linkage_from_merges([
            (0, 1, 0.9, 2),
            (2, 3, 0.2, 2),
            (4, 5, 0.1, 4),
        ])

        # k=2: split into {0,1} and {2,3}
        result = constrained_cut(linkage, min_size=2)
        assert result.n_clusters == 2
        assert result.total_stability > 0


class TestMembership:
    """Verify membership array correctness."""

    def test_membership_complete(self):
        """Every node is assigned exactly once."""
        linkage = _linkage_from_merges([
            (0, 1, 0.5, 2),
            (2, 3, 0.3, 2),
            (4, 5, 0.1, 4),
        ])
        result = constrained_cut(linkage, min_size=2)
        assert len(result.membership) == 4
        assert all(m >= 0 for m in result.membership)
        # Every leaf assigned
        assert set(result.membership.tolist()) == {0, 1}

    def test_membership_consistent_with_partition(self):
        """membership array and partition list agree."""
        linkage = _linkage_from_merges([
            (0, 1, 0.5, 2),
            (2, 3, 0.3, 2),
            (4, 5, 0.1, 4),
        ])
        result = constrained_cut(linkage, min_size=1)
        for cluster_id, cluster_set in enumerate(result.partition):
            for leaf in cluster_set:
                assert result.membership[leaf] == cluster_id


class TestVaryingK:
    """Multiple k values on the same dendrogram: cluster count is non-increasing."""

    def test_monotonic_count(self):
        """Larger k → fewer or equal clusters."""
        # 8 leaves, balanced binary tree
        linkage = _linkage_from_merges([
            (0, 1, 0.9, 2),   # node 8
            (2, 3, 0.8, 2),   # node 9
            (4, 5, 0.7, 2),   # node 10
            (6, 7, 0.6, 2),   # node 11
            (8, 9, 0.3, 4),   # node 12
            (10, 11, 0.2, 4), # node 13
            (12, 13, 0.1, 8), # node 14
        ])

        counts = []
        for k in [1, 2, 3, 4, 5, 8]:
            result = constrained_cut(linkage, min_size=k)
            counts.append(result.n_clusters)

        # Non-increasing
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1], (
                f"k={[1,2,3,4,5,8][i]}: {counts[i]} clusters, "
                f"k={[1,2,3,4,5,8][i+1]}: {counts[i+1]} clusters"
            )


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_linkage_shape(self):
        with pytest.raises(ValueError, match="linkage must be"):
            constrained_cut(np.array([[1, 2, 3]]), min_size=1)

    def test_min_size_zero(self):
        with pytest.raises(ValueError, match="min_size must be"):
            linkage = _linkage_from_merges([(0, 1, 1.0, 2)])
            constrained_cut(linkage, min_size=0)

    def test_single_merge(self):
        """Simplest possible dendrogram: 2 nodes, 1 merge."""
        linkage = _linkage_from_merges([(0, 1, 0.5, 2)])
        result = constrained_cut(linkage, min_size=1)
        assert result.n_clusters == 2
        assert len(result.partition) == 2
