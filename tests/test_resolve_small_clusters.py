"""Tests for resolve_small_clusters (contracted-graph adaptive CPM coarsening)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sciscape.clustering.postprocess import (
    resolve_small_clusters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(coarse_memberships: list[list[int]]) -> MagicMock:
    """Create a mock runner that supports contract → clone → run pipeline.

    *coarse_memberships* are membership vectors over **supernodes**
    (one element per unique cluster), not over original nodes.
    """
    runner = MagicMock()

    # contract() returns a mock contracted graph
    mock_graph = MagicMock()
    mock_graph.ecount.return_value = 10  # for log message
    runner.contract.return_value = mock_graph

    # clone_for_graph() returns a contracted runner
    contracted_runner = MagicMock()
    runner.clone_for_graph.return_value = contracted_runner

    # contracted_runner.run() returns supernode-level memberships
    results = []
    for mem in coarse_memberships:
        result = MagicMock()
        result.membership = mem
        results.append(result)
    contracted_runner.run.side_effect = results

    return runner


# ---------------------------------------------------------------------------
# Tests: no small clusters
# ---------------------------------------------------------------------------

class TestNoSmallClusters:
    def test_all_large(self):
        runner = _make_runner([])
        mem = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        result = resolve_small_clusters(runner, mem, gamma=0.001, min_size=3)

        assert result.membership == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert result.n_small_initial == 0
        runner.contract.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: basic resolution
# ---------------------------------------------------------------------------

class TestBasicResolution:
    def test_small_cluster_moves_to_dominant_large(self):
        # Cluster 0 (5 nodes), cluster 1 (5 nodes) = large; cluster 2 (2 nodes) = small
        fine_mem = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
        # Supernodes: 0(large), 1(large), 2(small)
        # At coarse γ: supernode 2 joins community with supernode 0
        coarse = [0, 1, 0]

        runner = _make_runner([coarse])
        result = resolve_small_clusters(runner, fine_mem, gamma=0.001, min_size=5)

        assert result.membership[10] == 0
        assert result.membership[11] == 0
        assert result.n_clusters_resolved == 1
        assert result.n_nodes_resolved == 2

    def test_cluster_moves_together(self):
        """All nodes in a small cluster must move to the same target."""
        fine_mem = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # All supernodes join same community
        coarse = [0, 0, 0]

        runner = _make_runner([coarse])
        result = resolve_small_clusters(runner, fine_mem, gamma=0.001, min_size=4)

        # Cluster 2 (3 nodes) should all go to same target
        assert result.membership[10] == result.membership[11] == result.membership[12]

    def test_multiple_small_different_targets(self):
        """Two small clusters resolve to different large clusters."""
        fine_mem = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3]
        # Supernodes: 0(large), 1(large), 2(small), 3(small)
        # Super 2 joins 0, super 3 joins 1
        coarse = [0, 1, 0, 1]

        runner = _make_runner([coarse])
        result = resolve_small_clusters(runner, fine_mem, gamma=0.001, min_size=5)

        assert result.membership[10] == 0
        assert result.membership[12] == 1
        assert result.n_clusters_resolved == 2
        assert result.n_nodes_resolved == 4


# ---------------------------------------------------------------------------
# Tests: adaptive multi-level convergence
# ---------------------------------------------------------------------------

class TestAdaptiveConvergence:
    def test_dense_resolved_early_sparse_later(self):
        """Dense-region small cluster resolves at mild coarsening,
        sparse-region one needs stronger coarsening."""
        fine_mem = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3]
        # Supernodes: 0(large), 1(large), 2(small), 3(small)

        # Level 1: super 2 joins 0, super 3 stays isolated (community 3)
        coarse1 = [0, 1, 0, 3]
        # Level 2: super 3 joins 1
        coarse2 = [0, 1, 0, 1]

        runner = _make_runner([coarse1, coarse2])
        result = resolve_small_clusters(runner, fine_mem, gamma=0.001, min_size=5)

        assert result.membership[10] == 0  # resolved at level 1
        assert result.membership[12] == 1  # resolved at level 2
        assert result.n_clusters_resolved == 2
        assert len(result.resolutions_used) == 2

    def test_stops_early_when_all_resolved(self):
        fine_mem = [0, 0, 0, 0, 0, 1, 1]
        # Supernodes: 0(large), 1(small)
        coarse1 = [0, 0]

        runner = _make_runner([coarse1])
        result = resolve_small_clusters(runner, fine_mem, gamma=0.001, min_size=5)

        assert result.n_clusters_resolved == 1
        assert len(result.resolutions_used) == 1
        contracted_runner = runner.clone_for_graph.return_value
        contracted_runner.run.assert_called_once()

    def test_convergence_stops_after_two_zero_rounds(self):
        """Stops after 2 consecutive rounds with zero resolutions."""
        fine_mem = [0, 0, 0, 0, 0, 1, 1]
        # Supernodes: 0(large), 1(small)
        # Super 1 never merges with 0 (stays in its own community)
        coarse1 = [0, 1]  # round 1: 0 resolved
        coarse2 = [0, 1]  # round 2: 0 resolved → converge

        runner = _make_runner([coarse1, coarse2])
        result = resolve_small_clusters(runner, fine_mem, gamma=0.001, min_size=5)

        assert result.n_clusters_unresolvable == 1
        assert len(result.resolutions_used) == 2
        contracted_runner = runner.clone_for_graph.return_value
        assert contracted_runner.run.call_count == 2

    def test_halving_gamma_each_round(self):
        """Each round halves the factor: γ*0.5, γ*0.25, γ*0.125, ..."""
        fine_mem = [0, 0, 0, 0, 0, 1, 1]
        coarse1 = [0, 1]  # 0 resolved
        coarse2 = [0, 1]  # 0 resolved → converge

        runner = _make_runner([coarse1, coarse2])
        resolve_small_clusters(runner, fine_mem, gamma=0.004, min_size=5)

        contracted_runner = runner.clone_for_graph.return_value
        calls = contracted_runner.run.call_args_list
        # Check gamma values (ignore other kwargs)
        assert calls[0].args[0] == pytest.approx(0.002)   # 0.004 * 0.5
        assert calls[1].args[0] == pytest.approx(0.001)   # 0.004 * 0.25


# ---------------------------------------------------------------------------
# Tests: unresolvable clusters
# ---------------------------------------------------------------------------

class TestUnresolvable:
    def test_small_clusters_isolated_from_large(self):
        fine_mem = [0, 0, 0, 0, 0, 1, 1, 2, 2]
        # Supernodes: 0(large), 1(small), 2(small)
        # Smalls group together but not with large
        coarse1 = [0, 1, 1]
        coarse2 = [0, 1, 1]

        runner = _make_runner([coarse1, coarse2])
        result = resolve_small_clusters(runner, fine_mem, gamma=0.001, min_size=5)

        assert result.n_clusters_unresolvable == 2
        assert result.n_nodes_unresolvable == 4
        assert result.membership[5] == 1  # unchanged
        assert result.membership[7] == 2  # unchanged


# ---------------------------------------------------------------------------
# Tests: contraction correctness
# ---------------------------------------------------------------------------

class TestContraction:
    def test_contract_called_with_correct_mapping(self):
        """Verify contraction uses 0-based supernode IDs."""
        fine_mem = [0, 0, 0, 0, 0, 1, 1]
        runner = _make_runner([[0, 0]])
        resolve_small_clusters(runner, fine_mem, gamma=0.001, min_size=5)

        runner.contract.assert_called_once()
        contracted_mem = runner.contract.call_args[0][0]
        # 5 nodes of cluster 0 → super 0, 2 nodes of cluster 1 → super 1
        assert contracted_mem == [0, 0, 0, 0, 0, 1, 1]

    def test_node_sizes_passed_to_run(self):
        """CPM node_sizes reflect original cluster sizes."""
        fine_mem = [0, 0, 0, 0, 0, 1, 1]
        runner = _make_runner([[0, 0]])
        resolve_small_clusters(runner, fine_mem, gamma=0.001, min_size=5)

        contracted_runner = runner.clone_for_graph.return_value
        call_kwargs = contracted_runner.run.call_args_list[0].kwargs
        assert call_kwargs["node_sizes"] == [5, 2]


# ---------------------------------------------------------------------------
# Tests: input safety
# ---------------------------------------------------------------------------

class TestInputSafety:
    def test_original_not_mutated(self):
        original = [0, 0, 0, 0, 0, 1, 1]
        frozen = list(original)
        coarse = [0, 0]

        runner = _make_runner([coarse])
        resolve_small_clusters(runner, original, gamma=0.001, min_size=5)

        assert original == frozen
