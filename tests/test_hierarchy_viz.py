"""Tests for sciscape.keyword_extraction.visualization._hierarchy."""

from __future__ import annotations

import pandas as pd
import pytest

from sciscape.keyword_extraction.visualization._hierarchy import _build_hierarchy_df


def _make_df(with_depth: bool = True) -> pd.DataFrame:
    rows = []
    for cid in range(3):
        for i in range(12):
            row = {
                "cluster_id": cid,
                "term": f"kw_{cid}_{i}",
                "score": round(1.0 - i * 0.07, 4),
                "frequency": 100 - i * 5,
            }
            if with_depth:
                row["depth_level"] = i % 3
            rows.append(row)
    return pd.DataFrame(rows)


class TestBuildHierarchyDf:
    def test_basic_structure(self):
        df = _make_df()
        hdf = _build_hierarchy_df(df, top_n_per_depth=5)
        assert "cluster" in hdf.columns
        assert "depth" in hdf.columns
        assert "term" in hdf.columns
        assert "score" in hdf.columns

    def test_respects_top_n(self):
        df = _make_df()
        hdf = _build_hierarchy_df(df, top_n_per_depth=2)
        # 3 clusters × 3 depth levels × 2 terms = max 18
        assert len(hdf) <= 18

    def test_no_depth_column(self):
        df = _make_df(with_depth=False)
        hdf = _build_hierarchy_df(df, top_n_per_depth=5)
        # All assigned to depth 0 → "Broad"
        assert set(hdf["depth"].unique()) == {"Broad"}

    def test_empty_df(self):
        df = pd.DataFrame(columns=["cluster_id", "term", "score", "frequency", "depth_level"])
        hdf = _build_hierarchy_df(df)
        assert len(hdf) == 0

    def test_cluster_labels(self):
        df = _make_df()
        hdf = _build_hierarchy_df(df)
        for label in hdf["cluster"].unique():
            assert label.startswith("C")
            assert ":" in label


def _has_plotly() -> bool:
    try:
        import plotly  # noqa: F401
        return True
    except ImportError:
        return False


class TestPlotImport:
    def test_importable(self):
        from sciscape.keyword_extraction.visualization import (
            plot_cluster_treemap,
            plot_cluster_sunburst,
        )
        assert callable(plot_cluster_treemap)
        assert callable(plot_cluster_sunburst)


@pytest.mark.skipif(not _has_plotly(), reason="plotly not installed")
class TestPlotTreemap:
    def test_returns_figure(self):
        import plotly.graph_objects as go
        from sciscape.keyword_extraction.visualization import plot_cluster_treemap
        fig = plot_cluster_treemap(_make_df())
        assert isinstance(fig, go.Figure)

    def test_empty_df(self):
        from sciscape.keyword_extraction.visualization import plot_cluster_treemap
        df = pd.DataFrame(columns=["cluster_id", "term", "score", "frequency", "depth_level"])
        fig = plot_cluster_treemap(df)
        assert fig is not None

    def test_color_by_cluster(self):
        from sciscape.keyword_extraction.visualization import plot_cluster_treemap
        fig = plot_cluster_treemap(_make_df(), color_by="cluster")
        assert fig is not None


@pytest.mark.skipif(not _has_plotly(), reason="plotly not installed")
class TestPlotSunburst:
    def test_returns_figure(self):
        import plotly.graph_objects as go
        from sciscape.keyword_extraction.visualization import plot_cluster_sunburst
        fig = plot_cluster_sunburst(_make_df())
        assert isinstance(fig, go.Figure)

    def test_no_depth(self):
        from sciscape.keyword_extraction.visualization import plot_cluster_sunburst
        fig = plot_cluster_sunburst(_make_df(with_depth=False))
        assert fig is not None
