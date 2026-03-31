"""Tests for gamma_search (optimised γ binary search with warm-start)."""

from __future__ import annotations

from unittest.mock import MagicMock


from sciscape.clustering.postprocess import (
    gamma_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(memberships: list[list[int]]) -> MagicMock:
    """Create a mock runner whose run() returns memberships in order.

    If fewer calls are made than memberships provided, extras are ignored.
    If more calls are made (e.g., refinement probes), the last membership
    is recycled via side_effect + default.
    """
    runner = MagicMock()
    results = []
    for mem in memberships:
        result = MagicMock()
        result.membership = mem
        results.append(result)
    runner.run.side_effect = results
    return runner


def _make_constant_runner(membership: list[int], n_calls: int = 20) -> MagicMock:
    """Runner that always returns the same membership."""
    return _make_runner([membership] * n_calls)


# ---------------------------------------------------------------------------
# Tests: basic search
# ---------------------------------------------------------------------------

class TestBasicSearch:
    def test_finds_best_gamma(self):
        """The search picks the γ that produces the most large clusters."""
        # 3 coarse probes: each returns different partition quality
        mem_lo = [0] * 10  # 1 cluster, n_large=0 (10 < 1000 but let's use min_size=5)
        mem_mid = [0] * 5 + [1] * 5  # 2 clusters of 5, n_large=2
        mem_hi = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 singletons, n_large=0

        runner = _make_runner([mem_lo, mem_mid, mem_hi] * 3)
        result = gamma_search(
            runner,
            gamma_range=(1e-6, 1e-3),
            min_size=5,
            max_refine=0,  # skip refinement to test coarse only
        )

        assert result.n_large == 2
        assert result.n_evals == 3

    def test_returns_correct_membership(self):
        """Returned membership matches the best γ's result."""
        mem = [0] * 6 + [1] * 4
        runner = _make_constant_runner(mem)
        result = gamma_search(
            runner,
            gamma_range=(1e-6, 1e-3),
            min_size=3,
            max_refine=0,
        )

        assert result.membership == mem


class TestWarmStart:
    def test_warm_start_passes_initial_membership(self):
        """With warm_start=True, probes after the first pass initial_membership."""
        mem = [0] * 10
        runner = _make_constant_runner(mem)
        gamma_search(
            runner,
            gamma_range=(1e-6, 1e-3),
            min_size=5,
            max_refine=0,
            warm_start=True,
        )

        calls = runner.run.call_args_list
        # First call has no initial_membership (no cache yet)
        assert "initial_membership" not in calls[0].kwargs
        # Subsequent calls should have initial_membership
        for c in calls[1:]:
            assert "initial_membership" in c.kwargs

    def test_no_warm_start(self):
        """With warm_start=False, no call passes initial_membership."""
        mem = [0] * 10
        runner = _make_constant_runner(mem)
        gamma_search(
            runner,
            gamma_range=(1e-6, 1e-3),
            min_size=5,
            max_refine=0,
            warm_start=False,
        )

        for c in runner.run.call_args_list:
            assert "initial_membership" not in c.kwargs


class TestSearchIterations:
    def test_passes_n_iterations(self):
        """search_iterations is forwarded as n_iterations."""
        mem = [0] * 10
        runner = _make_constant_runner(mem)
        gamma_search(
            runner,
            gamma_range=(1e-6, 1e-3),
            min_size=5,
            search_iterations=10,
            max_refine=0,
        )

        for c in runner.run.call_args_list:
            assert c.kwargs.get("n_iterations") == 10

    def test_none_iterations_uses_default(self):
        """search_iterations=None does not pass n_iterations."""
        mem = [0] * 10
        runner = _make_constant_runner(mem)
        gamma_search(
            runner,
            gamma_range=(1e-6, 1e-3),
            min_size=5,
            search_iterations=None,
            max_refine=0,
        )

        for c in runner.run.call_args_list:
            assert "n_iterations" not in c.kwargs


class TestRefinement:
    def test_refinement_adds_probes(self):
        """max_refine > 0 adds midpoint probes beyond the initial 3."""
        mem = [0] * 10
        runner = _make_constant_runner(mem)
        result = gamma_search(
            runner,
            gamma_range=(1e-6, 1e-3),
            min_size=5,
            max_refine=2,
        )

        # 3 coarse + at least 1 refinement probe
        assert result.n_evals >= 3

    def test_zero_refine_only_coarse(self):
        """max_refine=0 does exactly n_coarse evaluations."""
        mem = [0] * 10
        runner = _make_constant_runner(mem)
        result = gamma_search(
            runner,
            gamma_range=(1e-6, 1e-3),
            min_size=5,
            n_coarse=5,
            max_refine=0,
        )

        assert result.n_evals == 5


class TestCoarsePoints:
    def test_custom_n_coarse(self):
        """n_coarse controls how many initial probes."""
        mem = [0] * 10
        runner = _make_constant_runner(mem)
        result = gamma_search(
            runner,
            gamma_range=(1e-6, 1e-3),
            min_size=5,
            n_coarse=7,
            max_refine=0,
        )

        assert result.n_evals == 7
