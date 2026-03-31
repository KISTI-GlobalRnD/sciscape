"""Tests for sciscape.keyword_extraction.visualization._network_map."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sciscape.keyword_extraction.visualization._network_map import (
    _cluster_keyword_sets,
    _mds_layout,
    _pairwise_jaccard,
    _spring_layout,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n_clusters: int = 3, n_per_cluster: int = 10) -> pd.DataFrame:
    """Build a minimal keyword DataFrame."""
    rows = []
    for cid in range(n_clusters):
        for i in range(n_per_cluster):
            rows.append({
                "cluster_id": cid,
                "term": f"kw_{cid}_{i}" if i < n_per_cluster - 2 else f"shared_{i}",
                "score": round(1.0 - i * 0.08, 4),
                "frequency": 100 - i * 5,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _cluster_keyword_sets
# ---------------------------------------------------------------------------

class TestClusterKeywordSets:
    def test_returns_dict_of_sets(self):
        df = _make_df(2, 5)
        result = _cluster_keyword_sets(df)
        assert isinstance(result, dict)
        assert all(isinstance(v, set) for v in result.values())

    def test_correct_cluster_ids(self):
        df = _make_df(3, 4)
        result = _cluster_keyword_sets(df)
        assert set(result.keys()) == {0, 1, 2}

    def test_keyword_count(self):
        df = _make_df(2, 6)
        result = _cluster_keyword_sets(df)
        assert len(result[0]) == 6
        assert len(result[1]) == 6


# ---------------------------------------------------------------------------
# _pairwise_jaccard
# ---------------------------------------------------------------------------

class TestPairwiseJaccard:
    def test_identity_diagonal(self):
        df = _make_df(3, 5)
        kw_sets = _cluster_keyword_sets(df)
        cids, sim = _pairwise_jaccard(kw_sets)
        for i in range(len(cids)):
            assert sim[i, i] == pytest.approx(1.0)

    def test_symmetric(self):
        df = _make_df(3, 8)
        kw_sets = _cluster_keyword_sets(df)
        _, sim = _pairwise_jaccard(kw_sets)
        np.testing.assert_array_almost_equal(sim, sim.T)

    def test_shared_terms_increase_similarity(self):
        """Clusters sharing terms should have higher Jaccard than disjoint ones."""
        df = _make_df(3, 10)  # last 2 terms per cluster are "shared_8", "shared_9"
        kw_sets = _cluster_keyword_sets(df)
        _, sim = _pairwise_jaccard(kw_sets)
        # All pairs share the same 2 terms, so off-diagonal should be > 0
        assert sim[0, 1] > 0
        assert sim[0, 2] > 0

    def test_disjoint_clusters_zero_similarity(self):
        rows = [
            {"cluster_id": 0, "term": "alpha", "score": 1.0, "frequency": 10},
            {"cluster_id": 1, "term": "beta", "score": 1.0, "frequency": 10},
        ]
        df = pd.DataFrame(rows)
        kw_sets = _cluster_keyword_sets(df)
        _, sim = _pairwise_jaccard(kw_sets)
        assert sim[0, 1] == 0.0

    def test_empty_input(self):
        cids, sim = _pairwise_jaccard({})
        assert cids == []
        assert sim.shape == (0, 0)


# ---------------------------------------------------------------------------
# _mds_layout
# ---------------------------------------------------------------------------

class TestMDSLayout:
    def test_output_shape(self):
        sim = np.eye(4)
        pos = _mds_layout(sim)
        assert pos.shape == (4, 2)

    def test_single_cluster(self):
        sim = np.array([[1.0]])
        pos = _mds_layout(sim)
        assert pos.shape == (1, 2)

    def test_deterministic(self):
        rng = np.random.RandomState(7)
        sim = rng.rand(5, 5)
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 1.0)
        p1 = _mds_layout(sim, random_state=0)
        p2 = _mds_layout(sim, random_state=0)
        np.testing.assert_array_almost_equal(p1, p2)


# ---------------------------------------------------------------------------
# _spring_layout
# ---------------------------------------------------------------------------

class TestSpringLayout:
    def test_output_shape(self):
        sim = np.eye(3)
        pos = _spring_layout(sim)
        assert pos.shape == (3, 2)

    def test_bounded(self):
        rng = np.random.RandomState(0)
        sim = rng.rand(6, 6)
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 1.0)
        pos = _spring_layout(sim)
        assert np.abs(pos).max() <= 1.0 + 1e-6

    def test_deterministic(self):
        sim = np.array([[1.0, 0.5], [0.5, 1.0]])
        p1 = _spring_layout(sim, random_state=42)
        p2 = _spring_layout(sim, random_state=42)
        np.testing.assert_array_almost_equal(p1, p2)


# ---------------------------------------------------------------------------
# Integration: plot functions (import-only, no plotly required at test time)
# ---------------------------------------------------------------------------

def _has_plotly() -> bool:
    try:
        import plotly  # noqa: F401
        return True
    except ImportError:
        return False


class TestPlotFunctionsImport:
    """Verify the public API is importable."""

    def test_importable(self):
        from sciscape.keyword_extraction.visualization import (
            plot_cluster_map,
            plot_cluster_map_with_keywords,
        )
        assert callable(plot_cluster_map)
        assert callable(plot_cluster_map_with_keywords)


class TestPlotClusterMap:
    """Smoke tests — only run when plotly is available."""

    @pytest.fixture()
    def df(self):
        return _make_df(3, 10)

    @pytest.mark.skipif(
        not _has_plotly(), reason="plotly not installed"
    )
    def test_returns_figure(self, df):
        from sciscape.keyword_extraction.visualization import plot_cluster_map
        fig = plot_cluster_map(df)
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(
        not _has_plotly(), reason="plotly not installed"
    )
    def test_spring_layout(self, df):
        from sciscape.keyword_extraction.visualization import plot_cluster_map
        fig = plot_cluster_map(df, layout="spring")
        assert fig is not None

    @pytest.mark.skipif(
        not _has_plotly(), reason="plotly not installed"
    )
    def test_empty_df(self):
        from sciscape.keyword_extraction.visualization import plot_cluster_map
        df = pd.DataFrame(columns=["cluster_id", "term", "score", "frequency"])
        fig = plot_cluster_map(df)
        assert fig is not None
