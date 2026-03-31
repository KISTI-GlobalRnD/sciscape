"""Tests for split_large_clusters and refine_clusters."""

from __future__ import annotations

from collections import Counter
from unittest.mock import MagicMock


from sciscape.clustering.postprocess import (
    refine_clusters,
    split_large_clusters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_split_runner(
    sub_memberships: list[list[int]] | None = None,
) -> MagicMock:
    """Create a mock runner for split tests.

    *sub_memberships* — one per oversized cluster (sorted by cid).
    Each sub-runner always returns the same membership regardless of γ,
    so the γ search converges immediately.
    """
    runner = MagicMock()

    if sub_memberships is None:
        sub_memberships = []

    sub_runners = []
    for mem in sub_memberships:
        sub_runner = MagicMock()
        result = MagicMock()
        result.membership = mem
        sub_runner.run.return_value = result
        sub_runners.append(sub_runner)

    runner.clone_for_graph.side_effect = sub_runners
    return runner


def _make_refine_runner(
    sub_memberships: list[list[int]] | None = None,
    coarse_memberships: list[list[int]] | None = None,
) -> MagicMock:
    """Create a mock runner that supports both split and merge.

    For split: runner.graph.induced_subgraph → subgraph,
               runner.clone_for_graph → sub_runner
    For merge: runner.contract → contracted graph,
               runner.clone_for_graph → contracted_runner
    """
    runner = MagicMock()

    # Build a sequence of clone_for_graph return values:
    # split sub_runners first, then merge contracted_runners
    clones = []

    # Split sub-runners
    if sub_memberships:
        for mem in sub_memberships:
            sub_runner = MagicMock()
            result = MagicMock()
            result.membership = mem
            sub_runner.run.return_value = result
            clones.append(sub_runner)

    # Merge contracted-runners
    if coarse_memberships:
        mock_graph = MagicMock()
        mock_graph.ecount.return_value = 10
        runner.contract.return_value = mock_graph

        contracted_runner = MagicMock()
        results = []
        for mem in coarse_memberships:
            result = MagicMock()
            result.membership = mem
            results.append(result)
        contracted_runner.run.side_effect = results
        clones.append(contracted_runner)

    runner.clone_for_graph.side_effect = clones
    return runner


# ---------------------------------------------------------------------------
# Tests: no oversized clusters
# ---------------------------------------------------------------------------

class TestNoOversized:
    def test_all_within_limit(self):
        runner = _make_split_runner()
        mem = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        result = split_large_clusters(runner, mem, gamma=0.001, min_size=3)

        assert result.membership == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert result.n_clusters_split == 0
        assert result.n_new_clusters_created == 0
        runner.clone_for_graph.assert_not_called()

    def test_exact_threshold_not_split(self):
        """Cluster of size exactly 2*min_size is NOT split (need > max_size)."""
        runner = _make_split_runner()
        mem = [0] * 6 + [1] * 6  # size 6 each, max_size = 2*3 = 6
        result = split_large_clusters(runner, mem, gamma=0.001, min_size=3)

        assert result.n_clusters_split == 0


# ---------------------------------------------------------------------------
# Tests: basic split
# ---------------------------------------------------------------------------

class TestBasicSplit:
    def test_single_oversized_splits(self):
        """One oversized cluster (10 nodes) splits into two sub-clusters."""
        # Cluster 0: 10 nodes (> 2*3=6), Cluster 1: 3 nodes
        mem = [0] * 10 + [1] * 3
        # Subgraph of cluster 0 → splits into [0,0,0,0,0, 1,1,1,1,1]
        sub_mem = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        runner = _make_split_runner([sub_mem])
        result = split_large_clusters(runner, mem, gamma=0.001, min_size=3)

        assert result.n_clusters_split == 1
        assert result.n_nodes_affected == 10
        # Original ID kept for largest sub-cluster (both are size 5)
        sizes = Counter(result.membership)
        # Should have 3 clusters now
        assert len(sizes) == 3
        # Cluster 1 (3 nodes) unchanged
        assert result.membership[10] == 1

    def test_largest_sub_keeps_original_id(self):
        """The largest sub-cluster retains the original cluster ID."""
        # Cluster 0: 8 nodes (> 2*3=6)
        mem = [0] * 8
        # Split: 5+3 → sub 0 has 5 nodes (largest), sub 1 has 3
        sub_mem = [0, 0, 0, 0, 0, 1, 1, 1]

        runner = _make_split_runner([sub_mem])
        result = split_large_clusters(runner, mem, gamma=0.001, min_size=3)

        # The 5-node sub-cluster should keep ID 0
        assert result.membership[:5] == [0, 0, 0, 0, 0]
        # The 3-node sub-cluster gets a new ID
        new_id = result.membership[5]
        assert new_id != 0
        assert result.membership[5:8] == [new_id] * 3

    def test_multiple_oversized(self):
        """Two oversized clusters both split."""
        # Cluster 0: 8 nodes, Cluster 1: 8 nodes, Cluster 2: 3 nodes
        mem = [0] * 8 + [1] * 8 + [2] * 3
        # Cluster 0 splits into [0,0,0,0, 1,1,1,1]
        sub_mem_0 = [0, 0, 0, 0, 1, 1, 1, 1]
        # Cluster 1 splits into [0,0,0,0,0, 1,1,1]
        sub_mem_1 = [0, 0, 0, 0, 0, 1, 1, 1]

        runner = _make_split_runner([sub_mem_0, sub_mem_1])
        result = split_large_clusters(runner, mem, gamma=0.001, min_size=3)

        assert result.n_clusters_split == 2
        sizes = Counter(result.membership)
        assert len(sizes) == 5  # 2 original each split into 2, + cluster 2


# ---------------------------------------------------------------------------
# Tests: unsplittable clusters
# ---------------------------------------------------------------------------

class TestUnsplittable:
    def test_no_meaningful_split(self):
        """γ search finds n_large < 2 → cluster kept intact."""
        # Cluster 0: 10 nodes (oversized)
        mem = [0] * 10
        # Subgraph always returns single cluster → n_large = 1
        sub_mem = [0] * 10

        runner = _make_split_runner([sub_mem])
        result = split_large_clusters(runner, mem, gamma=0.001, min_size=3)

        assert result.n_clusters_split == 0
        assert result.membership == [0] * 10

    def test_split_into_tiny_pieces_rejected(self):
        """If all sub-clusters are below min_size, n_large=0 < 2 → no split."""
        mem = [0] * 10
        # Splits into 5 pairs: sizes all = 2 < min_size=3 → n_large=0
        sub_mem = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

        runner = _make_split_runner([sub_mem])
        result = split_large_clusters(runner, mem, gamma=0.001, min_size=3)

        assert result.n_clusters_split == 0
        assert result.membership == [0] * 10


# ---------------------------------------------------------------------------
# Tests: new ID uniqueness
# ---------------------------------------------------------------------------

class TestNewIds:
    def test_new_ids_dont_conflict(self):
        """New cluster IDs must not collide with existing IDs."""
        # Cluster 0: 10 nodes, Cluster 5: 3 nodes (gap in IDs)
        mem = [0] * 10 + [5] * 3
        sub_mem = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        runner = _make_split_runner([sub_mem])
        result = split_large_clusters(runner, mem, gamma=0.001, min_size=3)

        all_ids = set(result.membership)
        # New ID should be max(existing)+1 = 6
        assert 6 in all_ids
        # Cluster 5 unchanged
        assert all(result.membership[i] == 5 for i in range(10, 13))

    def test_multiple_splits_sequential_ids(self):
        """Multiple new sub-clusters get sequential IDs."""
        # Cluster 0: 12 nodes → splits into 3 sub-clusters
        mem = [0] * 12
        sub_mem = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

        runner = _make_split_runner([sub_mem])
        result = split_large_clusters(runner, mem, gamma=0.001, min_size=3)

        assert result.n_new_clusters_created == 2  # 1 keeps original, 2 new
        all_ids = set(result.membership)
        assert len(all_ids) == 3


# ---------------------------------------------------------------------------
# Tests: input safety
# ---------------------------------------------------------------------------

class TestSplitInputSafety:
    def test_original_not_mutated(self):
        original = [0] * 10 + [1] * 3
        frozen = list(original)
        sub_mem = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        runner = _make_split_runner([sub_mem])
        split_large_clusters(runner, original, gamma=0.001, min_size=3)

        assert original == frozen

    def test_explicit_max_size(self):
        """Explicit max_size overrides the 2*min_size default."""
        mem = [0] * 8 + [1] * 3
        sub_mem = [0, 0, 0, 0, 0, 1, 1, 1]

        runner = _make_split_runner([sub_mem])
        # min_size=3 → default max_size=6, but explicit max_size=10
        result = split_large_clusters(
            runner, mem, gamma=0.001, min_size=3, max_size=10,
        )
        # 8 <= 10, so no split
        assert result.n_clusters_split == 0


# ---------------------------------------------------------------------------
# Tests: refine_clusters (split + merge loop)
# ---------------------------------------------------------------------------

class TestRefineNoChanges:
    def test_all_good_sizes(self):
        """No oversized or undersized → single round, no changes."""
        runner = MagicMock()
        mem = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        # No oversized, no small → both phases do nothing

        # Mock contract for merge phase (returns graph with no edges needed)
        mock_graph = MagicMock()
        mock_graph.ecount.return_value = 0
        runner.contract.return_value = mock_graph

        result = refine_clusters(runner, mem, gamma=0.001, min_size=3)

        assert result.membership == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert result.n_rounds == 1
        assert result.split_results[0].n_clusters_split == 0
        assert result.merge_results[0].n_clusters_resolved == 0


class TestRefineConvergence:
    def test_stops_when_stable(self):
        """Loop stops when neither split nor merge makes changes."""
        runner = MagicMock()
        mem = [0, 0, 0, 1, 1, 1]

        mock_graph = MagicMock()
        mock_graph.ecount.return_value = 0
        runner.contract.return_value = mock_graph

        result = refine_clusters(
            runner, mem, gamma=0.001, min_size=3, max_rounds=5,
        )

        # Should stop after 1 round since nothing to do
        assert result.n_rounds == 1
