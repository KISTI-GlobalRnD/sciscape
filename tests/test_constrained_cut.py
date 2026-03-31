"""Tests for size-constrained optimal cut on dendrograms."""

from __future__ import annotations

import numpy as np
import pytest

from sciscape.clustering.constrained_cut import constrained_cut


# ---------------------------------------------------------------------------
# Helper: build simple linkage matrices for testing
# ---------------------------------------------------------------------------

def _linkage_from_merges(merges: list[tuple[int, int, float, int]]) -> np.ndarray:
    """Build a similarity linkage matrix from a list of merges.

    Each merge is (left_id, right_id, height, subtree_size).
    Heights should be non-increasing (CPM density convention).
    """
    return np.array(merges, dtype=np.float64)


# ---------------------------------------------------------------------------
# Test fixtures: simple dendrograms
# ---------------------------------------------------------------------------

class TestTrivial:
    """Edge cases and trivial dendrograms."""

    def test_two_nodes_k1(self):
        """Two nodes, k=1 → 2 singleton clusters."""
        linkage = _linkage_from_merges([(0, 1, 1.0, 2)])
        result = constrained_cut(linkage, min_size=1)
        assert result.n_clusters == 2
        assert result.feasible is True
        assert len(result.partition) == 2
        assert result.membership[0] != result.membership[1]

    def test_two_nodes_k2(self):
        """Two nodes, k=2 → 1 cluster (entire tree)."""
        linkage = _linkage_from_merges([(0, 1, 1.0, 2)])
        result = constrained_cut(linkage, min_size=2)
        assert result.n_clusters == 1
        assert result.feasible is True
        assert set().union(*result.partition) == {0, 1}

    def test_two_nodes_k3(self):
        """Two nodes, k=3 → infeasible, returns single cluster fallback."""
        linkage = _linkage_from_merges([(0, 1, 1.0, 2)])
        result = constrained_cut(linkage, min_size=3)
        assert result.n_clusters == 1
        assert result.feasible is False


class TestBalancedTree:
    """Balanced binary tree with 4 leaves."""

    @pytest.fixture()
    def linkage_4(self):
        """
        Dendrogram (similarity, non-increasing heights):
                  6 (h=0.1, size=4)
                 / \\
               4     5
            (h=0.5) (h=0.3)
            / \\    / \\
           0   1  2   3

        Merge order (highest ρ first):
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
        assert result.feasible is True

    def test_k2_gives_2_clusters(self, linkage_4):
        """k=2 → {0,1} and {2,3}."""
        result = constrained_cut(linkage_4, min_size=2)
        assert result.n_clusters == 2
        assert result.feasible is True
        clusters = [frozenset(c) for c in result.partition]
        assert frozenset({0, 1}) in clusters
        assert frozenset({2, 3}) in clusters

    def test_k3_gives_1_cluster(self, linkage_4):
        """k=3 → cannot split (each subtree has only 2), keep all as one."""
        result = constrained_cut(linkage_4, min_size=3)
        assert result.n_clusters == 1
        assert result.feasible is True
        assert set().union(*result.partition) == {0, 1, 2, 3}

    def test_k4_gives_1_cluster(self, linkage_4):
        """k=4 → exactly the root."""
        result = constrained_cut(linkage_4, min_size=4)
        assert result.n_clusters == 1
        assert result.feasible is True


class TestUnbalancedTree:
    """Unbalanced tree to test asymmetric splitting."""

    @pytest.fixture()
    def linkage_unbalanced(self):
        """
        5 leaves, chain-like tree:
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
        """k=2 → singletons are too small, can only keep root as 1 cluster."""
        result = constrained_cut(linkage_unbalanced, min_size=2)
        assert result.n_clusters == 1

    def test_k1_maximizes_to_5(self, linkage_unbalanced):
        """k=1 → all 5 singletons."""
        result = constrained_cut(linkage_unbalanced, min_size=1)
        assert result.n_clusters == 5


class TestStabilityTiebreak:
    """Tie-breaking by total stability (persistence)."""

    def test_prefer_higher_stability(self):
        """Two different cut strategies give different stabilities."""
        linkage = _linkage_from_merges([
            (0, 1, 0.9, 2),
            (2, 3, 0.2, 2),
            (4, 5, 0.1, 4),
        ])

        # k=2: split into {0,1} and {2,3}
        result = constrained_cut(linkage, min_size=2)
        assert result.n_clusters == 2
        assert result.total_stability > 0
        assert result.feasible is True


class TestFeasibility:
    """Test the feasible flag in CutResult."""

    def test_feasible_when_satisfiable(self):
        linkage = _linkage_from_merges([(0, 1, 1.0, 2)])
        result = constrained_cut(linkage, min_size=1)
        assert result.feasible is True

    def test_infeasible_when_too_large_k(self):
        linkage = _linkage_from_merges([(0, 1, 1.0, 2)])
        result = constrained_cut(linkage, min_size=10)
        assert result.feasible is False
        assert result.n_clusters == 1  # fallback


class TestDisconnectedDendrogram:
    """Test with height-0 merges (disconnected components in the graph)."""

    def test_zero_height_merges(self):
        """Two components merged at density 0."""
        # Component 1: 0+1 at ρ=1.0
        # Component 2: 2+3 at ρ=1.0
        # Bridge: merge at ρ=0.0
        linkage = _linkage_from_merges([
            (0, 1, 1.0, 2),
            (2, 3, 1.0, 2),
            (4, 5, 0.0, 4),
        ])
        result = constrained_cut(linkage, min_size=2)
        assert result.n_clusters == 2
        assert result.feasible is True
        clusters = [frozenset(c) for c in result.partition]
        assert frozenset({0, 1}) in clusters
        assert frozenset({2, 3}) in clusters

    def test_all_zero_height(self):
        """All merges at density 0 (fully disconnected graph)."""
        linkage = _linkage_from_merges([
            (0, 1, 0.0, 2),
            (2, 3, 0.0, 2),
            (4, 5, 0.0, 4),
        ])
        result = constrained_cut(linkage, min_size=2)
        assert result.n_clusters == 2
        assert result.feasible is True


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
        assert result.feasible is True
        assert len(result.partition) == 2

    def test_single_node(self):
        """Single node: empty linkage, min_size=1."""
        linkage = np.empty((0, 4), dtype=np.float64)
        result = constrained_cut(linkage, min_size=1, n_leaves=1)
        assert result.n_clusters == 1
        assert result.feasible is True
        assert 0 in result.partition[0]


class TestLeafSizes:
    """Tests for leaf_sizes parameter (contracted graph / supernode support)."""

    def test_leaf_sizes_enables_split(self):
        """With leaf_sizes, supernodes satisfy min_size that leaves alone can't."""
        # 3 supernodes: sizes [500, 800, 700]
        # Dendrogram (built with node_sizes): merge 0+1 → 1300, then +2 → 2000
        linkage = _linkage_from_merges([
            (0, 1, 0.5, 1300),  # node 3 = {0,1}, 500+800
            (3, 2, 0.1, 2000),  # node 4 = {0,1,2}, 1300+700
        ])
        leaf_sizes = np.array([500, 800, 700])

        # With leaf_sizes, min_size=600:
        #   leaf 0: 500 < 600 → infeasible
        #   leaf 1: 800 ≥ 600 → 1 cluster
        #   leaf 2: 700 ≥ 600 → 1 cluster
        #   node 3 ({0,1}): split? leaf 0 infeasible. keep? 1300≥600 → 1 cluster
        #   root: split? node3=1, leaf2=1 → split gives 2 clusters
        result = constrained_cut(linkage, min_size=600, leaf_sizes=leaf_sizes)
        assert result.n_clusters == 2
        assert result.feasible is True

    def test_leaf_sizes_forces_merge(self):
        """Large min_size forces small supernodes to merge."""
        # 3 supernodes: sizes [100, 200, 150]
        linkage = _linkage_from_merges([
            (0, 1, 0.5, 300),   # node 3 = {0,1}, 100+200
            (3, 2, 0.1, 450),   # node 4 = {0,1,2}, 300+150
        ])
        leaf_sizes = np.array([100, 200, 150])

        # min_size=250: leaf 2 (150) too small, {0,1}=300 ok
        # Split would need both children feasible, but leaf 2=150 < 250
        # So only option is keep root as 1 cluster (450 ≥ 250)
        result = constrained_cut(linkage, min_size=250, leaf_sizes=leaf_sizes)
        assert result.n_clusters == 1
        assert result.feasible is True

    def test_leaf_sizes_none_equals_default(self):
        """leaf_sizes=None behaves identically to leaf_sizes=[1,1,...,1]."""
        linkage = _linkage_from_merges([
            (0, 1, 0.5, 2),
            (2, 3, 0.3, 2),
            (4, 5, 0.1, 4),
        ])
        result_none = constrained_cut(linkage, min_size=2)
        result_ones = constrained_cut(
            linkage, min_size=2,
            leaf_sizes=np.array([1, 1, 1, 1]),
        )
        assert result_none.n_clusters == result_ones.n_clusters
        np.testing.assert_array_equal(result_none.membership, result_ones.membership)

    def test_leaf_sizes_validation(self):
        """leaf_sizes with wrong length raises ValueError."""
        linkage = _linkage_from_merges([(0, 1, 1.0, 2)])
        with pytest.raises(ValueError, match="leaf_sizes length"):
            constrained_cut(linkage, min_size=1, leaf_sizes=np.array([10]))
