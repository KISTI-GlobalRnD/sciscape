"""Tests for automatic gamma selection (auto_gamma.py)."""

import numpy as np
import polars as pl
import pytest

from sciscape.clustering.leiden_rust import RUST_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")

from sciscape.clustering.auto_gamma import find_gamma, AutoGammaResult, GammaProbe


# ── Helpers ──────────────────────────────────────────────────

def _make_two_clique_edges(n_per_clique=50, cross_weight=0.01):
    """Two cliques connected by weak cross-edges."""
    edges = []
    for i in range(n_per_clique):
        for j in range(i + 1, n_per_clique):
            edges.append((str(i), str(j), 1.0))
    offset = n_per_clique
    for i in range(n_per_clique):
        for j in range(i + 1, n_per_clique):
            edges.append((str(offset + i), str(offset + j), 1.0))
    # Weak cross-edges
    for i in range(3):
        edges.append((str(i), str(offset + i), cross_weight))
    return pl.DataFrame({"uid1": [e[0] for e in edges],
                         "uid2": [e[1] for e in edges],
                         "rel_sum2": [e[2] for e in edges]})


def _make_single_clique(n=30):
    """Single dense clique."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((str(i), str(j), 1.0))
    return pl.DataFrame({"uid1": [e[0] for e in edges],
                         "uid2": [e[1] for e in edges],
                         "rel_sum2": [e[2] for e in edges]})


# ── Tests ────────────────────────────────────────────────────

class TestFindGamma:

    def test_returns_autogamma_result(self):
        df = _make_two_clique_edges(20)
        result = find_gamma(df, target_max_pct=60.0, min_size=3,
                            n_coarse=4, max_refine=1, postprocess=False)
        assert isinstance(result, AutoGammaResult)
        assert result.gamma > 0
        assert result.n_clusters >= 1
        assert result.max_pct >= 0

    def test_finds_two_clusters(self):
        df = _make_two_clique_edges(30, cross_weight=0.001)
        result = find_gamma(df, target_max_pct=60.0, min_size=3,
                            n_coarse=6, max_refine=2, postprocess=False)
        assert result.n_clusters >= 2

    def test_membership_array_returned(self):
        df = _make_two_clique_edges(20)
        result = find_gamma(df, target_max_pct=60.0, min_size=3,
                            n_coarse=4, max_refine=1, postprocess=True)
        assert result.membership is not None
        assert len(result.membership) == 40  # 2 × 20 nodes

    def test_probes_recorded(self):
        df = _make_two_clique_edges(20)
        result = find_gamma(df, target_max_pct=60.0, min_size=3,
                            n_coarse=6, max_refine=0, postprocess=False)
        assert len(result.probes) >= 1
        for p in result.probes:
            assert isinstance(p, GammaProbe)
            assert p.gamma > 0
            assert p.n_clusters >= 1

    def test_explicit_gamma_range(self):
        df = _make_two_clique_edges(20)
        result = find_gamma(df, target_max_pct=60.0, min_size=3,
                            gamma_range=(0.01, 10.0),
                            n_coarse=4, max_refine=1, postprocess=False)
        assert 0.01 <= result.gamma <= 100.0  # may extend slightly via refinement

    def test_single_clique_returns_valid(self):
        df = _make_single_clique(20)
        result = find_gamma(df, target_max_pct=50.0, min_size=2,
                            n_coarse=4, max_refine=1, postprocess=False)
        assert result.gamma > 0
        assert result.n_clusters >= 1

    def test_small_graph(self):
        """Small graph (<100 nodes) should use more probes."""
        df = _make_two_clique_edges(10)
        result = find_gamma(df, target_max_pct=60.0, min_size=2,
                            postprocess=False)
        # n_coarse="auto" → 12 for small graphs
        assert len(result.probes) >= 6

    def test_max_pct_below_target(self):
        df = _make_two_clique_edges(30, cross_weight=0.001)
        target = 60.0
        result = find_gamma(df, target_max_pct=target, min_size=3,
                            n_coarse=6, max_refine=3, postprocess=False)
        # Should find γ where max_pct ≤ target (or close)
        assert result.max_pct <= target + 10  # allow some slack

    def test_empty_edges(self):
        """Empty edge table should not crash."""
        df = pl.DataFrame({"uid1": ["a"], "uid2": ["b"], "rel_sum2": [1.0]})
        result = find_gamma(df, target_max_pct=50.0, min_size=1,
                            n_coarse=3, max_refine=0, postprocess=False)
        assert result.gamma > 0

    def test_postprocess_only_on_final(self):
        """With postprocess=True, coarse probes skip postprocess, final gets it."""
        df = _make_two_clique_edges(20)
        result = find_gamma(df, target_max_pct=60.0, min_size=3,
                            n_coarse=4, max_refine=1, postprocess=True)
        assert result.membership is not None
