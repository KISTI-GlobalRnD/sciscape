"""Tests for sciscape.keyword_extraction.visualization._temporal."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sciscape.keyword_extraction.visualization._temporal import (
    _extract_year_matrix,
    _parse_year_series,
)


def _make_df(n_clusters: int = 2, n_per: int = 8) -> pd.DataFrame:
    rows = []
    for cid in range(n_clusters):
        for i in range(n_per):
            series = {str(y): max(1, 10 - abs(y - 2020) - i) for y in range(2015, 2025)}
            rows.append({
                "cluster_id": cid,
                "term": f"kw_{cid}_{i}",
                "score": round(1.0 - i * 0.1, 4),
                "frequency": 50 - i * 3,
                "pub_year_series": json.dumps(series),
            })
    return pd.DataFrame(rows)


class TestParseYearSeries:
    def test_json_string(self):
        s = pd.Series([json.dumps({"2020": 5})])
        result = _parse_year_series(s)
        assert result.iloc[0] == {"2020": 5}

    def test_dict(self):
        s = pd.Series([{"2020": 5}])
        result = _parse_year_series(s)
        assert result.iloc[0] == {"2020": 5}

    def test_none(self):
        s = pd.Series([None])
        result = _parse_year_series(s)
        assert result.iloc[0] == {}

    def test_invalid_json(self):
        s = pd.Series(["not json"])
        result = _parse_year_series(s)
        assert result.iloc[0] == {}


class TestExtractYearMatrix:
    def test_basic(self):
        df = _make_df()
        mat, terms, years = _extract_year_matrix(df, top_n=5)
        assert len(terms) == 5
        assert len(years) > 0
        assert mat.shape == (5, len(years))

    def test_no_metric_column(self):
        df = pd.DataFrame({"cluster_id": [0], "term": ["a"], "score": [1.0]})
        mat, terms, years = _extract_year_matrix(df)
        assert mat.empty

    def test_top_n_limit(self):
        df = _make_df(n_per=20)
        mat, terms, years = _extract_year_matrix(df, top_n=3)
        assert len(terms) == 3


def _has_plotly() -> bool:
    try:
        import plotly  # noqa: F401
        return True
    except ImportError:
        return False


class TestPlotImport:
    def test_importable(self):
        from sciscape.keyword_extraction.visualization import (
            plot_temporal_heatmap,
            plot_cluster_trend_comparison,
        )
        assert callable(plot_temporal_heatmap)
        assert callable(plot_cluster_trend_comparison)


@pytest.mark.skipif(not _has_plotly(), reason="plotly not installed")
class TestPlotTemporalHeatmap:
    def test_returns_figure(self):
        import plotly.graph_objects as go
        from sciscape.keyword_extraction.visualization import plot_temporal_heatmap
        fig = plot_temporal_heatmap(_make_df())
        assert isinstance(fig, go.Figure)

    def test_single_cluster(self):
        from sciscape.keyword_extraction.visualization import plot_temporal_heatmap
        fig = plot_temporal_heatmap(_make_df(), cluster_id=0)
        assert fig is not None

    def test_normalize(self):
        from sciscape.keyword_extraction.visualization import plot_temporal_heatmap
        fig = plot_temporal_heatmap(_make_df(), normalize_rows=True)
        assert fig is not None

    def test_no_data(self):
        from sciscape.keyword_extraction.visualization import plot_temporal_heatmap
        df = pd.DataFrame({"cluster_id": [0], "term": ["a"], "score": [1.0]})
        fig = plot_temporal_heatmap(df)
        assert fig is not None


@pytest.mark.skipif(not _has_plotly(), reason="plotly not installed")
class TestPlotClusterTrendComparison:
    def test_returns_figure(self):
        import plotly.graph_objects as go
        from sciscape.keyword_extraction.visualization import plot_cluster_trend_comparison
        fig = plot_cluster_trend_comparison(_make_df())
        assert isinstance(fig, go.Figure)

    def test_mean_aggregate(self):
        from sciscape.keyword_extraction.visualization import plot_cluster_trend_comparison
        fig = plot_cluster_trend_comparison(_make_df(), aggregate="mean")
        assert fig is not None

    def test_no_data(self):
        from sciscape.keyword_extraction.visualization import plot_cluster_trend_comparison
        df = pd.DataFrame({"cluster_id": [0], "term": ["a"], "score": [1.0]})
        fig = plot_cluster_trend_comparison(df)
        assert fig is not None
