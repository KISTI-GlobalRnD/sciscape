"""Tests for resolve_singletons (hierarchical CPM singleton inheritance)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sciscape.clustering.postprocess import (
    resolve_singletons,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(coarse_memberships: list[list[int]]) -> MagicMock:
    """Create a mock LeidenRunner that returns pre-defined memberships."""
    runner = MagicMock()
    results = []
    for mem in coarse_memberships:
        result = MagicMock()
        result.membership = mem
        results.append(result)
    runner.run.side_effect = results
    return runner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResolveSingletonsNoSingletons:
    """When there are no singletons, nothing changes."""

    def test_no_singletons(self):
        runner = _make_runner([])
        mem = [0, 0, 0, 1, 1, 1]
        result = resolve_singletons(runner, mem, gamma=0.001)

        assert result.membership == [0, 0, 0, 1, 1, 1]
        assert result.n_singletons_initial == 0
        assert result.n_resolved == 0
        assert result.n_unresolvable == 0
        runner.run.assert_not_called()


class TestResolveSingletonsBasic:
    """Singleton gets assigned to dominant fine cluster in its coarse group."""

    def test_single_singleton_resolved(self):
        # Fine: node 0 = singleton (cluster 99), nodes 1-5 = cluster 0
        fine_mem = [99, 0, 0, 0, 0, 0]
        # Coarse: node 0 joins same group as nodes 1-5 (all cluster 0)
        coarse_mem = [0, 0, 0, 0, 0, 0]

        runner = _make_runner([coarse_mem])
        result = resolve_singletons(runner, fine_mem, gamma=0.001,
                                    coarsening_factors=(0.1,))

        assert result.membership[0] == 0  # singleton assigned to cluster 0
        assert result.n_singletons_initial == 1
        assert result.n_resolved == 1
        assert result.n_unresolvable == 0

    def test_singleton_picks_dominant_cluster(self):
        # Fine: node 0 = singleton, nodes 1-2 = cluster A, nodes 3-5 = cluster B
        fine_mem = [99, 1, 1, 2, 2, 2]
        # Coarse: all in one group → singleton should pick cluster 2 (dominant)
        coarse_mem = [0, 0, 0, 0, 0, 0]

        runner = _make_runner([coarse_mem])
        result = resolve_singletons(runner, fine_mem, gamma=0.001,
                                    coarsening_factors=(0.1,))

        assert result.membership[0] == 2  # cluster B is dominant (3 vs 2)

    def test_multiple_singletons(self):
        # Two singletons: nodes 0 and 6; cluster 1 has nodes 4,5 (size 2)
        fine_mem = [10, 0, 0, 0, 1, 1, 11]
        # Coarse: node 0 → group 0 (with cluster 0); node 6 → group 1 (with cluster 1)
        coarse_mem = [0, 0, 0, 0, 1, 1, 1]

        runner = _make_runner([coarse_mem])
        result = resolve_singletons(runner, fine_mem, gamma=0.001,
                                    coarsening_factors=(0.1,))

        assert result.membership[0] == 0
        assert result.membership[6] == 1
        assert result.n_resolved == 2


class TestResolveSingletonsTwoLevels:
    """Singletons not resolved at first coarse level get resolved at second."""

    def test_second_level_resolves_remaining(self):
        # Fine: nodes 0,1 are singletons; rest in cluster 2
        fine_mem = [10, 11, 2, 2, 2, 2]
        # First coarse: node 0 joins cluster 2's group; node 1 stays isolated
        coarse1 = [0, 5, 0, 0, 0, 0]
        # Second coarse: node 1 also joins the group
        coarse2 = [0, 0, 0, 0, 0, 0]

        runner = _make_runner([coarse1, coarse2])
        result = resolve_singletons(runner, fine_mem, gamma=0.001,
                                    coarsening_factors=(0.1, 0.01))

        assert result.membership[0] == 2  # resolved at level 1
        assert result.membership[1] == 2  # resolved at level 2
        assert result.n_resolved == 2
        assert result.n_unresolvable == 0
        assert len(result.resolutions_used) == 2

    def test_skips_second_level_if_all_resolved(self):
        fine_mem = [10, 0, 0, 0]
        coarse1 = [0, 0, 0, 0]

        runner = _make_runner([coarse1])
        result = resolve_singletons(runner, fine_mem, gamma=0.001,
                                    coarsening_factors=(0.1, 0.01))

        assert result.n_resolved == 1
        assert len(result.resolutions_used) == 1  # only 1 run needed
        runner.run.assert_called_once()


class TestResolveSingletonsUnresolvable:
    """Singletons that remain isolated at all levels stay unresolved."""

    def test_singleton_only_coarse_cluster(self):
        # Two singletons end up alone in their coarse cluster (no non-singleton peers)
        fine_mem = [10, 11, 0, 0, 0]
        coarse1 = [5, 5, 0, 0, 0]   # nodes 0,1 group together but both singletons
        coarse2 = [5, 5, 0, 0, 0]   # still isolated

        runner = _make_runner([coarse1, coarse2])
        result = resolve_singletons(runner, fine_mem, gamma=0.001,
                                    coarsening_factors=(0.1, 0.01))

        # Singletons can't resolve because their coarse cluster has no non-singleton
        assert result.n_unresolvable == 2
        assert result.membership[0] == 10  # unchanged
        assert result.membership[1] == 11  # unchanged


class TestResolveSingletonsOriginalUnchanged:
    """The original membership list is not mutated."""

    def test_input_not_mutated(self):
        original = [99, 0, 0, 0]
        frozen_copy = list(original)
        coarse = [0, 0, 0, 0]

        runner = _make_runner([coarse])
        resolve_singletons(runner, original, gamma=0.001,
                           coarsening_factors=(0.1,))

        assert original == frozen_copy  # input unchanged


class TestResolveSingletonsGammaValues:
    """Correct gamma values are passed to the runner."""

    def test_gamma_factors(self):
        fine_mem = [99, 0, 0, 0]
        coarse = [0, 0, 0, 0]

        runner = _make_runner([coarse])
        result = resolve_singletons(runner, fine_mem, gamma=0.002,
                                    coarsening_factors=(0.1,))

        runner.run.assert_called_once_with(pytest.approx(0.0002))
        assert result.resolutions_used == [pytest.approx(0.0002)]
